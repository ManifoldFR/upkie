#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import numpy as np
import pinocchio as pin
import hppfcl as fcl


def box_position_limits(model: pin.Model) -> Tuple[np.ndarray, np.ndarray]:
    r"""!Compute position limits in box format:

    \f[
    q_{min} \leq q \leq q_{max}
    \f]

    @param model Pinocchio model.
    @returns Tuple ``(q_min, q_max)`` of lower and upper position limits, with
    -infinity and +infinity where there is no limit.
    """
    no_position_limit = np.logical_or(
        model.upperPositionLimit > 1e20,
        model.upperPositionLimit < model.lowerPositionLimit + 1e-10,
    )
    q_min = model.lowerPositionLimit.copy()
    q_max = model.upperPositionLimit.copy()
    q_min[no_position_limit] = -np.inf
    q_max[no_position_limit] = +np.inf
    return q_min, q_max


def box_velocity_limits(model: pin.Model) -> Tuple[np.ndarray, np.ndarray]:
    r"""!Compute velocity limits in box format:

    \f[
    v_{min} \leq v \leq v_{max}
    \f]

    @param model Pinocchio model.
    @return Velocity limits, with -infinity and +infinity where there is no
    limit.
    """
    no_velocity_limit = np.logical_or(
        model.velocityLimit > 1e20,
        model.velocityLimit < 1e-10,
    )
    v_max = model.velocityLimit.copy()
    v_max[no_velocity_limit] = np.inf
    return v_max


def box_torque_limits(model: pin.Model) -> Tuple[np.ndarray, np.ndarray]:
    r"""!Compute velocity limits in box format:

    \f[
    \tau_{min} \leq \tau \leq \tau_{max}
    \f]

    @param model Pinocchio model.
    @return Torque limits, with -infinity and +infinity where there is no
    limit.
    """
    no_torque_limit = np.logical_or(
        model.effortLimit > 1e20,
        model.effortLimit < 1e-10,
    )
    tau_max = model.effortLimit.copy()
    tau_max[no_torque_limit] = np.inf
    return tau_max


def create_cartpole(N) -> Tuple[pin.Model, pin.GeometryModel]:
    """Create a cartpole Pinocchio model."""
    model = pin.Model()
    geom_model = pin.GeometryModel()

    parent_id = 0

    cart_radius = 0.1
    cart_length = 5 * cart_radius
    cart_mass = 1.0
    joint_name = "joint_cart"

    geometry_placement = pin.SE3.Identity()
    geometry_placement.rotation = pin.Quaternion(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    ).toRotationMatrix()

    joint_id = model.addJoint(
        parent_id, pin.JointModelPY(), pin.SE3.Identity(), joint_name
    )

    body_inertia = pin.Inertia.FromCylinder(
        cart_mass, cart_radius, cart_length
    )
    body_placement = geometry_placement
    model.appendBodyToJoint(joint_id, body_inertia, body_placement)
    # We need to rotate the inertia as it is expressed in the LOCAL
    # frame of the geometry

    shape_cart = fcl.Cylinder(cart_radius, cart_length)

    geom_cart = pin.GeometryObject(
        "shape_cart", joint_id, geometry_placement, shape_cart
    )
    geom_cart.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_cart)

    parent_id = joint_id
    joint_placement = pin.SE3.Identity()
    body_mass = 0.1
    body_radius = 0.1
    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(
            parent_id, pin.JointModelRX(), joint_placement, joint_name
        )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = 1.0
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = fcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(
            geom1_name, joint_id, body_placement, shape1
        )
        geom1_obj.meshColor = np.ones((4))
        geom_model.addGeometryObject(geom1_obj)

        geom2_name = "bar_" + str(k + 1)
        shape2 = fcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(
            geom2_name, joint_id, shape2_placement, shape2
        )
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        parent_id = joint_id
        joint_placement = body_placement.copy()
    end_frame = pin.Frame(
        "end_effector_frame",
        model.getJointId("joint_" + str(N)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    geom_model.collision_pairs = []
    model.qinit = np.zeros(model.nq)
    model.qinit[1] = 0.0 * np.pi
    model.qref = pin.neutral(model)
    return model, geom_model
