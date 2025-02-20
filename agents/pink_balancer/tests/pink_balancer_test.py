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

"""Test main balancer functions."""

import unittest

from agents.pink_balancer.pink_balancer import parse_command_line_arguments


class TestPinkBalancer(unittest.TestCase):
    def test_no_config_argument(self):
        with self.assertRaises(SystemExit):
            parse_command_line_arguments()  # no --config


if __name__ == "__main__":
    unittest.main()
