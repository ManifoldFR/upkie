#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria

"""Wheel balancing using model predictive control of an LTV system."""

import asyncio
import os
import sys
import time

import gymnasium as gym
import mpacklog
import numpy as np
from ltv_mpc import solve_mpc
from ltv_mpc.systems import CartPole

import upkie.envs
from upkie.utils.clamp import clamp_and_warn
from upkie.utils.filters import low_pass_filter
from upkie.utils.raspi import on_raspi
from upkie.utils.spdlog import logging

upkie.envs.register()


def get_target_states(
    cart_pole: CartPole, state: np.ndarray, target_vel: float
):
    """Define the reference state trajectory over the receding horizon.

    Args:
        state: Cart-pole state at the beginning of the horizon.
        target_vel: Target ground velocity in m/s.

    Returns:
        Goal state at the end of the horizon.
    """
    nx = CartPole.STATE_DIM
    T = cart_pole.sampling_period
    target_states = np.zeros((cart_pole.nb_timesteps + 1) * nx)
    for k in range(cart_pole.nb_timesteps + 1):
        target_states[k * nx] = state[0] + (k * T) * target_vel
        target_states[k * nx + 2] = target_vel
    return target_states


async def balance(env: gym.Env, logger: mpacklog.AsyncLogger):
    """!
    Run proportional balancer in gym environment with logging.

    @param env Gym environment to Upkie.
    @param logger Additional logger.
    """
    cart_pole = CartPole(
        length=0.4,
        max_ground_accel=10.0,
        nb_timesteps=12,
        sampling_period=env.dt,
    )
    mpc_problem = cart_pole.build_mpc_problem(
        terminal_cost_weight=10.0,
        stage_state_cost_weight=1.0,
        stage_input_cost_weight=1e-3,
    )

    live_plot = None
    if not on_raspi():
        from ltv_mpc.live_plots import CartPolePlot  # imports matplotlib

        live_plot = CartPolePlot(cart_pole, order="velocities")

    env.reset()  # connects to the spine
    action = np.zeros(env.action_space.shape)
    commanded_velocity = 0.0
    while True:
        action[0] = commanded_velocity
        observation, _, terminated, truncated, info = await env.async_step(
            action
        )
        if terminated or truncated:
            observation, info = env.reset()

        observation_dict = info["observation"]
        ground_contact = observation_dict["floor_contact"]["contact"]

        # Unpack observation into initial MPC state
        (
            base_pitch,
            ground_position,
            base_angular_velocity,
            ground_velocity,
        ) = observation
        initial_state = np.array(
            [
                ground_position,
                base_pitch,
                ground_velocity,
                base_angular_velocity,
            ]
        )
        target_vel = 0.0
        target_states = get_target_states(cart_pole, initial_state, target_vel)

        mpc_problem.update_initial_state(initial_state)
        mpc_problem.update_goal_state(target_states[-CartPole.STATE_DIM :])
        mpc_problem.update_target_states(target_states[: -CartPole.STATE_DIM])

        plan = solve_mpc(mpc_problem, solver="proxqp")
        if not ground_contact:
            logging.info("Waiting for ground contact")
            commanded_velocity = low_pass_filter(
                prev_output=commanded_velocity,
                cutoff_period=0.1,
                new_input=0.0,
                dt=env.dt,
            )
        elif plan.is_empty:
            logging.error("Solver found no solution to the MPC problem")
            logging.info("Continuing with previous action")
        else:  # plan was found
            cart_pole.state = initial_state
            if live_plot is not None:
                t = time.time()
                live_plot.update(plan, t, initial_state, t)
            commanded_accel = plan.first_input
            commanded_velocity = clamp_and_warn(
                commanded_velocity + commanded_accel * env.dt,
                lower=-1.0,
                upper=+1.0,
                label="commanded_velocity",
            )
        await logger.put(  # log info to be written to file later
            {
                "action": action,
                "observation": info["observation"],
                "time": time.time(),
            }
        )
    await logger.stop()


async def main():
    """Main function of our asyncio program."""
    logger = mpacklog.AsyncLogger("mpc_balancing.mpack")
    with gym.make("UpkieWheelsEnv-v4", frequency=200.0) as env:
        await asyncio.gather(
            balance(env, logger),
            logger.write(),  # write logs to file when there is time
        )


if __name__ == "__main__":
    # TODO(scaron): move to a function in utils.raspi
    if on_raspi() and os.geteuid() != 0:
        print("Re-running as root so that we can set CPU affinity")
        args = ["sudo", "-E", sys.executable] + sys.argv + [os.environ]
        os.execlpe("sudo", *args)
    if on_raspi():
        CPUID = 3
        os.sched_setaffinity(0, {CPUID})

    asyncio.run(main())
