# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import json
import os
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from legged_lab.utils.virtual_joystick import VirtualJoystickServer, open_joystick_window


class VirtualJoystickServerTest(unittest.TestCase):
    def setUp(self):
        self.server = VirtualJoystickServer(port=0, max_vx=2.0, max_vy=1.0, max_wz=1.5, timeout=0.05)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def post(self, payload):
        request = urllib.request.Request(
            f"{self.server.url}/api/command",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return json.load(response)

    def post_reset(self):
        request = urllib.request.Request(f"{self.server.url}/api/reset", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return json.load(response)

    def test_page_and_state_are_served(self):
        with urllib.request.urlopen(self.server.url, timeout=1.0) as response:
            page = response.read().decode()
        self.assertIn("机器人目标速度", page)
        self.assertIn("/api/command", page)
        for key_code in ("KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE"):
            self.assertIn(key_code, page)
        self.assertIn("keyboardRiseRate = 1.25", page)
        self.assertIn("keyboardFallRate = 3.5", page)
        self.assertIn("requestAnimationFrame(updateKeyboard)", page)
        self.assertIn("setInterval(send, 20)", page)
        self.assertIn("键盘控制", page)
        self.assertNotIn('id="linear"', page)
        self.assertNotIn('id="turn"', page)

        with urllib.request.urlopen(f"{self.server.url}/api/state", timeout=1.0) as response:
            state = json.load(response)
        self.assertEqual(state["limits"], {"vx": 2.0, "vy": 1.0, "wz": 1.5})
        self.assertFalse(state["active"])

    def test_command_is_clamped_and_times_out_to_zero(self):
        state = self.post({"vx": 7.0, "vy": -4.0, "wz": 0.75})
        self.assertEqual(self.server.command(), (2.0, -1.0, 0.75))
        self.assertTrue(state["active"])

        time.sleep(0.07)
        self.assertEqual(self.server.command(), (0.0, 0.0, 0.0))

    def test_zero_limit_disables_an_axis(self):
        server = VirtualJoystickServer(port=0, max_vx=1.0, max_vy=0.0, max_wz=0.0)
        server.update({"vx": 0.5, "vy": 1.0, "wz": -1.0})
        self.assertEqual(server.command(), (0.5, 0.0, 0.0))

    def test_non_finite_command_is_rejected(self):
        request = urllib.request.Request(
            f"{self.server.url}/api/command",
            data=b'{"vx": NaN, "vy": 0, "wz": 0}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=1.0)
        self.assertEqual(caught.exception.code, 400)

    def test_reset_is_consumed_once_and_stops_command(self):
        self.post({"vx": 1.0, "vy": 0.2, "wz": -0.3})
        self.post_reset()
        self.assertEqual(self.server.command(), (0.0, 0.0, 0.0))
        self.assertTrue(self.server.consume_reset())
        self.assertFalse(self.server.consume_reset())


class OpenJoystickWindowTest(unittest.TestCase):
    @mock.patch("legged_lab.utils.virtual_joystick.subprocess.Popen")
    @mock.patch("legged_lab.utils.virtual_joystick.shutil.which", return_value="/usr/bin/google-chrome")
    def test_chrome_uses_desktop_app_mode(self, _, popen):
        mode = open_joystick_window("http://127.0.0.1:8765")
        self.assertEqual(str(mode), "isolated desktop window")
        command = popen.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/google-chrome")
        self.assertIn("--app=http://127.0.0.1:8765", command)
        profile_arg = next(arg for arg in command if arg.startswith("--user-data-dir="))
        profile_dir = profile_arg.split("=", 1)[1]
        self.assertTrue(os.path.isdir(profile_dir))
        popen.return_value.poll.return_value = 0
        mode.close()
        self.assertFalse(os.path.exists(profile_dir))

    @mock.patch("legged_lab.utils.virtual_joystick.webbrowser.open")
    @mock.patch("legged_lab.utils.virtual_joystick.shutil.which", return_value=None)
    def test_default_browser_is_fallback(self, _, browser_open):
        mode = open_joystick_window("http://127.0.0.1:8765")
        self.assertEqual(str(mode), "default browser")
        browser_open.assert_called_once_with("http://127.0.0.1:8765", new=2)


if __name__ == "__main__":
    unittest.main()
