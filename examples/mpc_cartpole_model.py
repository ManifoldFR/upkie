"""MPC on upkie using a Cartpole model and the proxddp trajopt library.

@authors Wilson Jallet, Quentin Le Lidec
@copyright 2023 Inria
"""
import pinocchio as pin
import numpy as np
import proxddp
import matplotlib.pyplot as plt
import argparse
import time
import gym
import asyncio
import mpacklog
import upkie.envs

from utils.pinocchio import create_cartpole
from pinocchio.visualize import MeshcatVisualizer
from proxddp import constraints, manifolds

upkie.envs.register()


def _parse_args():
    parser = argparse.ArgumentParser(
        "mpc_cartpole", description="Use nonlinear MPC to control the robot."
    )
    parser.add_argument("--bounds", action="store_true")
    parser.add_argument("--term_cstr", action="store_true")
    return parser.parse_args()


args = _parse_args()

model, geom_model = create_cartpole(1, cart_mass=1e-5, body_mass=5.37322)
visual_model = geom_model.clone()
data = model.createData()
geom_data = geom_model.createData()
nq = model.nq
nv = model.nv
space = manifolds.MultibodyPhaseSpace(model)
nx = space.nx
ndx = space.ndx
nu = 1
frame_id = model.getFrameId("end_effector_frame")


class BalancingTask:
    """Balancing task."""

    def __init__(self, dt: float):
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.dt = dt
        self.x0 = space.neutral()
        self.problem = self.create_problem()
        self.problem.setNumThreads(2)

    def create_problem(self) -> proxddp.TrajOptProblem:
        """Create a trajectory optimization problem."""
        act_mat = np.zeros((2, nu))
        act_mat[0, 0] = 1.0
        dt = self.dt
        self.cont_dyn = proxddp.dynamics.MultibodyFreeFwdDynamics(
            space, act_mat
        )
        self.dyn_model = proxddp.dynamics.IntegratorSemiImplEuler(
            self.cont_dyn, dt
        )

        # running cost regularizes the control input
        rcost = proxddp.CostStack(space, nu)
        wx = np.ones(ndx) * 1.0
        wu = np.ones(nu) * 1e-3
        rcost.addCost(
            proxddp.QuadraticStateCost(
                space, nu, self.x0, np.diag(wx) * dt
            )
        )
        rcost.addCost(
            proxddp.QuadraticControlCost(
                space, np.zeros(nu), np.diag(wu) * dt
            )
        )
        self.frame_place_target = pin.SE3.Identity()
        self.frame_place_target.translation[:] = self.target_pos
        term_cost = proxddp.CostStack(space, nu)

        # box constraint on control
        self.u_min = -6.0 * np.ones(nu)
        self.u_max = +6.0 * np.ones(nu)

        def get_box_cstr():
            """Get box constraint."""
            ctrl_fn = proxddp.ControlErrorResidual(ndx, nu)
            return proxddp.StageConstraint(
                ctrl_fn, constraints.BoxConstraint(self.u_min, self.u_max)
            )

        self.Tf = 1.0
        self.nsteps = int(self.Tf / self.dt)
        problem = proxddp.TrajOptProblem(self.x0, nu, space, term_cost)

        for i in range(self.nsteps):
            stage = proxddp.StageModel(rcost, self.dyn_model)
            if args.bounds:
                box_cstr = get_box_cstr()
                stage.addConstraint(box_cstr)
            problem.addStage(stage)

        term_fun = proxddp.FrameTranslationResidual(
            ndx, nu, model, self.target_pos, frame_id
        )

        if args.term_cstr:
            term_cstr = proxddp.StageConstraint(
                term_fun, constraints.EqualityConstraintSet()
            )
            problem.addTerminalConstraint(term_cstr)
        else:
            term_cost.addCost(
                proxddp.QuadraticStateCost(space, nu, self.x0, np.diag(wx) * dt)
            )
        return problem

    def create_initial_guess(self):
        """!Create an initial guess for the solver."""
        nsteps = self.nsteps
        u0 = np.zeros(nu)
        us_i = [u0] * nsteps
        xs_i = proxddp.rollout(self.dyn_model, self.x0, us_i)
        return xs_i, us_i


def create_solver(problem, tol=1e-4, add_history=False):
    mu_init = 1e-6
    verbose = proxddp.VerboseLevel.VERBOSE
    solver = proxddp.SolverProxDDP(tol, mu_init, verbose=verbose)
    solver.max_refinement_steps = 0
    if add_history:
        _callback = proxddp.HistoryCallback()
        solver.registerCallback("his", _callback)

    solver.setup(problem)
    workspace = solver.workspace
    for i in range(problem.num_steps):
        psc = workspace.getConstraintScaler(i)
        if args.bounds:
            psc.set_weight(10.0, 1)
    return solver


def update_problem(solver, task: BalancingTask, new_x0):
    task.problem.x0_init = new_x0
    ws: proxddp.Workspace = solver.workspace
    ws.cycleLeft()


def call_solver(solver, problem, xs_init, us_init, max_iter=1) -> proxddp.Results:
    solver.max_iters = max_iter
    solver.run(problem, xs_init, us_init)
    return solver.results


# perform the initial solve
task = BalancingTask(dt=0.01)
nsteps = task.nsteps
dt = task.dt
Tf = task.Tf
TOL = 1e-8


def create_plots(solver, res):
    """Some plots.

    TODO: Rearrange this to send to logging or something.
    """
    from proxddp.utils.plotting import plot_convergence

    u_min = task.u_min
    u_max = task.u_max

    fig1 = plt.figure(figsize=(7.2, 5.4))

    xs_opt = np.asarray(res.xs)
    trange = np.linspace(0, Tf, nsteps + 1)

    gs = plt.GridSpec(2, 1)
    gs0 = gs[0].subgridspec(1, 2)

    def get_endpoint(model, data: pin.Data, x, fid):
        nq = model.nq
        q = x[:nq]
        pin.framesForwardKinematics(model, data, q)
        return data.oMf[fid].translation.copy()

    def get_endpoint_traj(model, data, xs, fid):
        return [get_endpoint(model, data, x, fid) for x in xs]

    _pts = get_endpoint_traj(model, data, xs_opt, frame_id)
    _pts = _pts[:, 1:]

    ax1 = fig1.add_subplot(gs0[0])
    ax2 = fig1.add_subplot(gs0[1])
    lstyle = {"lw": 0.9}
    ax1.plot(trange, xs_opt[:, 0], ls="-", **lstyle)
    ax1.plot(trange, xs_opt[:, 2], ls="-", label="$\\dot{x}$", **lstyle)
    ax1.set_ylabel("$q(t)$")
    if args.term_cstr:
        pass
    ax1.legend()
    ax2.plot(trange, xs_opt[:, 1], ls="-", **lstyle)
    ax2.plot(trange, xs_opt[:, 3], ls="-", label="$\\dot{\\theta}$", **lstyle)
    ax2.set_ylabel("Angle $\\theta(t)$")
    ax2.legend()

    plt.xlabel("Time $t$")

    gs1 = gs[1].subgridspec(1, 2, width_ratios=[1, 2])
    ax3 = plt.subplot(gs1[0])
    plt.plot(*_pts.T, ls=":")
    plt.scatter(
        *task.target_pos[1:], c="r", marker="^", zorder=2, label="EE target"
    )
    plt.legend()
    ax3.set_aspect("equal")
    plt.title("Endpoint trajectory")

    plt.subplot(gs1[1])
    plt.plot(trange[:-1], res.us, label="$u(t)$", **lstyle)
    if args.bounds:
        plt.hlines(
            np.concatenate([u_min, u_max]),
            *trange[[0, -1]],
            ls="-",
            colors="k",
            lw=2.5,
            alpha=0.4,
            label=r"$\bar{u}$",
        )
    plt.title("Controls $u(t)$")
    plt.legend()
    fig1.tight_layout()

    fig2 = plt.figure(figsize=(6.4, 4.8))
    ax: plt.Axes = plt.subplot(111)
    ax.hlines(TOL, 0, res.num_iters, lw=2.2, alpha=0.8, colors="k")
    _callback = solver.getCallback("his")
    assert isinstance(_callback, proxddp.HistoryCallback)
    plot_convergence(_callback, ax, res)
    prim_tols = np.array(_callback.storage.prim_tols)
    al_iters = np.array(_callback.storage.al_iters)

    itrange = np.arange(len(al_iters))
    legends_ = [
        "$\\epsilon_\\mathrm{tol}$",
        "Prim. err $p$",
        "Dual err $d$",
    ]
    if len(itrange) > 0:
        ax.step(itrange, prim_tols, c="green", alpha=0.9, lw=1.1)
        al_change = al_iters[1:] - al_iters[:-1]
        al_change_idx = itrange[:-1][al_change > 0]
        legends_.extend(
            [
                "Prim tol $\\eta_k$",
                "AL iters",
            ]
        )

        ax.vlines(
            al_change_idx, *ax.get_ylim(), colors="gray", lw=4.0, alpha=0.5
        )
    ax.legend(
        [
            "$\\epsilon_\\mathrm{tol}$",
            "Prim. err $p$",
            "Dual err $d$",
            "Prim tol $\\eta_k$",
            "AL iters",
        ]
    )
    fig2.tight_layout()

    fig_dict = {"traj": fig1, "conv": fig2}

    TAG = "cartpole"
    if args.bounds:
        TAG += "_bounds"
    if args.term_cstr:
        TAG += "_cstr"

    for name, fig in fig_dict.items():
        fig.savefig(f"assets/{TAG}_{name}.png")
        fig.savefig(f"assets/{TAG}_{name}.pdf")

    plt.show()


def display_results(res: proxddp.Results):
    import hppfcl

    cp = [2.0, 0.0, 0.8]
    qs = [x[:nq] for x in res.xs.tolist()]
    vs = [x[nq:] for x in res.xs.tolist()]

    obj = pin.GeometryObject(
        "objective", 0, task.frame_place_target, hppfcl.Sphere(0.05)
    )
    color = [255, 20, 83, 255]
    obj.meshColor[:] = color
    obj.meshColor /= 255
    visual_model.addGeometryObject(obj)

    vizer = MeshcatVisualizer(
        model, geom_model, visual_model, data=data, collision_data=geom_data
    )
    vizer.initViewer(open=args.display, loadModel=True)
    vizer.setBackgroundColor()

    if args.record:
        fps = 1.0 / dt
        filename = "examples/ur5_reach_ctrlbox.mp4"
        ctx = vizer.create_video_ctx(filename, fps=fps)
        print(f"[video will be recorded @ {filename}]")
    else:
        from contextlib import nullcontext

        ctx = nullcontext()
        print("[no recording]")

    def _callback(i: int):
        pin.forwardKinematics(model, vizer.data, qs[i], vs[i])
        vizer.drawFrameVelocities(frame_id, color=0xFF9F1A)

    input("[Press enter]")
    with ctx:
        vizer.setCameraPosition(cp)
        vizer.play(qs, dt, _callback)


def nocopy_roll_insert(x, a=None):
    import copy
    if a is None:
        a = copy.copy(x[0])
    tail = x[1:]
    x[:-1] = tail
    x[-1] = a


async def mpc_balancer(env: gym.Env, logger: mpacklog.AsyncLogger):
    """Run proportional balancer in gym environment with logging."""
    observation = env.reset()  # connects to the spine
    action = np.zeros(env.action_space.shape)
    env_dt = 1. / env.frequency
    print("ENV DT: {}".format(env_dt))

    state = np.zeros(space.nx)
    tau = np.zeros(nu)
    gspeed = np.zeros(nu)

    mpc_dt = 0.015
    task = BalancingTask(mpc_dt)
    solver = create_solver(task.problem)
    solver.force_initial_condition = True
    _xsi, _usi = task.create_initial_guess()
    # initial solve
    res = call_solver(solver, task.problem, _xsi, _usi, max_iter=10)
    print(res)

    if env_dt <= mpc_dt:
        mpc_cycle_every = int(mpc_dt / env_dt)
    else:
        mpc_cycle_every = 1
    K0_fb = np.zeros((nu, ndx))
    dx_ref = np.zeros(ndx)
    t_since_mpc = 0.
    Kff = -1e-4

    for step in range(10_000):
        observation, _, done, info = await env.async_step(action)
        th, r, r_dot, th_dot = observation
        state[:] = r, th, r_dot, th_dot

        if step % mpc_cycle_every == 0:
            _xsi[0] = state
            update_problem(solver, task, state)
            res = call_solver(solver, task.problem, _xsi, _usi, 2)
            K0_fb[:] = res.controlFeedbacks()[0]
            t_since_mpc = 0.
            # update warm starts
            _xsi = res.xs.tolist()
            _usi = res.us.tolist()

        if done:
            observation = env.reset()
            _xsi, _usi = task.create_initial_guess()
            gspeed[:] = 0.

        x0_opt = res.xs[0]
        u0_opt = res.us[0]
        space.difference(x0_opt, state, dx_ref)
        tau[:] = u0_opt
        tau += K0_fb @ dx_ref
        gspeed += Kff * tau
        action[0] = gspeed  # 1D action: [ground_velocity]

        await logger.put(  # log info to be written to file later
            {
                "action": info["action"],
                "observation": info["observation"],
                "time": time.time(),
                "t_since_mpc": t_since_mpc
            }
        )
        t_since_mpc += env_dt


async def main_loop():
    """Main function of our asyncio program."""
    logger = mpacklog.AsyncLogger("wheeled_balancing.mpack")
    with gym.make("UpkieWheelsEnv-v3", frequency=200.0) as env:
        await asyncio.gather(
            mpc_balancer(env, logger),
            logger.write(),  # write logs to file when there is time
        )

if __name__ == "__main__":
    asyncio.run(main_loop())

