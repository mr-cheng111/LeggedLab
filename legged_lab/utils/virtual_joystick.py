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

"""Local desktop joystick for playback velocity commands."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <title>机器人目标速度</title>
  <style>
    * { box-sizing: border-box; }
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; background: #f2f4f3; color: #1e2421; touch-action: none; }
    main { width: min(760px, 100%); margin: 0 auto; padding: 22px 18px 28px; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-bottom: 1px solid #cbd1cd; padding-bottom: 14px; }
    h1 { margin: 0; font-size: 22px; font-weight: 680; letter-spacing: 0; }
    .status { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #53605a; white-space: nowrap; }
    .dot { width: 9px; height: 9px; border-radius: 50%; background: #aeb7b2; }
    .status.live .dot { background: #198754; box-shadow: 0 0 0 3px #d9eee3; }
    .values { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; margin-top: 18px; border: 1px solid #cbd1cd; border-radius: 6px; overflow: hidden; background: #cbd1cd; }
    .value { min-width: 0; padding: 12px 10px; text-align: center; background: #fff; }
    .value strong { display: block; font: 650 20px/1.2 ui-monospace, monospace; letter-spacing: 0; }
    .value span { display: block; margin-top: 4px; color: #68736d; font-size: 12px; }
    .keyboard { margin-top: 24px; padding: 18px; border: 1px solid #cbd1cd; border-radius: 6px; background: #fff; text-align: center; }
    .keyboard strong { display: block; margin-bottom: 8px; font-size: 16px; }
    .keyboard span { color: #53605a; font-size: 14px; line-height: 1.7; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 30px; }
    .actions button { width: 100%; height: 54px; border-radius: 6px; font-size: 16px; font-weight: 750; letter-spacing: 0; cursor: pointer; }
    .reset { border: 1px solid #78837d; background: #fff; color: #27312c; }
    .reset:active { background: #e5e9e7; }
    .stop { border: 1px solid #b4232c; background: #c92a35; color: #fff; }
    .stop:active { background: #a91d27; }
    .limits { margin: 12px 0 0; color: #68736d; font-size: 12px; text-align: center; }
  </style>
</head>
<body>
<main>
  <header><h1>机器人目标速度</h1><div id="status" class="status"><i class="dot"></i><span>正在连接</span></div></header>
  <section class="values" aria-label="目标速度">
    <div class="value"><strong id="vx">0.00</strong><span>vx m/s</span></div>
    <div class="value"><strong id="vy">0.00</strong><span>vy m/s</span></div>
    <div class="value"><strong id="wz">0.00</strong><span>wz rad/s</span></div>
  </section>
  <section class="keyboard"><strong>键盘控制</strong><span>W / S 前进后退　A / D 左右横移　Q / E 左右转向</span></section>
  <div class="actions">
    <button id="reset" class="reset" type="button">复位机器人</button>
    <button id="stop" class="stop" type="button">急停</button>
  </div>
  <p id="limits" class="limits"></p>
</main>
<script>
(() => {
  const command = { vx: 0, vy: 0, wz: 0 };
  let limits = { vx: 1, vy: 1, wz: 1 };
  let sendPending = false;
  let controlsReady = false;
  const pressed = new Set();
  const controlKeys = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);
  const keyboardAxis = { vx: 0, vy: 0, wz: 0 };
  const keyboardRiseRate = 1.25;
  const keyboardFallRate = 3.5;
  let keyboardActive = false;
  let keyboardFrameTime = performance.now();
  const $ = id => document.getElementById(id);
  const updateReadout = () => Object.keys(command).forEach(k => $(k).textContent = command[k].toFixed(2));
  const showStatus = live => {
    $('status').classList.toggle('live', live);
    $('status').querySelector('span').textContent = live ? '连接正常' : '连接断开';
  };
  async function send() {
    if (sendPending) return;
    sendPending = true;
    try {
      const response = await fetch('/api/command', {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(command)
      });
      showStatus(response.ok);
    } catch (_) { showStatus(false); }
    finally { sendPending = false; }
  }
  const keyboardTarget = () => ({
    vx: (pressed.has('KeyW') ? 1 : 0) - (pressed.has('KeyS') ? 1 : 0),
    vy: (pressed.has('KeyA') ? 1 : 0) - (pressed.has('KeyD') ? 1 : 0),
    wz: (pressed.has('KeyQ') ? 1 : 0) - (pressed.has('KeyE') ? 1 : 0)
  });
  const moveTowards = (value, target, amount) => {
    if (value < target) return Math.min(value + amount, target);
    if (value > target) return Math.max(value - amount, target);
    return value;
  };
  const beginKeyboardControl = () => {
    if (keyboardActive || !controlsReady) return;
    keyboardAxis.vx = limits.vx > 0 ? command.vx / limits.vx : 0;
    keyboardAxis.vy = limits.vy > 0 ? command.vy / limits.vy : 0;
    keyboardAxis.wz = limits.wz > 0 ? command.wz / limits.wz : 0;
    keyboardFrameTime = performance.now();
    keyboardActive = true;
  };
  const updateKeyboard = now => {
    const dt = Math.min(Math.max((now - keyboardFrameTime) / 1000, 0), 0.05);
    keyboardFrameTime = now;
    if (keyboardActive && controlsReady) {
      const target = keyboardTarget();
      for (const axis of Object.keys(keyboardAxis)) {
        const braking = target[axis] === 0 || keyboardAxis[axis] * target[axis] < 0;
        keyboardAxis[axis] = moveTowards(
          keyboardAxis[axis], target[axis], (braking ? keyboardFallRate : keyboardRiseRate) * dt
        );
      }
      command.vx = keyboardAxis.vx * limits.vx;
      command.vy = keyboardAxis.vy * limits.vy;
      command.wz = keyboardAxis.wz * limits.wz;
      if (pressed.size === 0 && Object.values(keyboardAxis).every(value => Math.abs(value) < 1e-4)) {
        keyboardAxis.vx = keyboardAxis.vy = keyboardAxis.wz = 0;
        command.vx = command.vy = command.wz = 0;
        keyboardActive = false;
        send();
      }
      updateReadout();
    }
    requestAnimationFrame(updateKeyboard);
  };
  const stop = () => {
    pressed.clear();
    keyboardActive = false;
    keyboardAxis.vx = keyboardAxis.vy = keyboardAxis.wz = 0;
    command.vx = command.vy = command.wz = 0;
    updateReadout(); send();
  };
  window.addEventListener('keydown', event => {
    if (!controlKeys.has(event.code)) return;
    event.preventDefault(); beginKeyboardControl(); pressed.add(event.code);
  });
  window.addEventListener('keyup', event => {
    if (!controlKeys.has(event.code)) return;
    event.preventDefault(); pressed.delete(event.code);
  });
  $('stop').addEventListener('click', stop);
  $('reset').addEventListener('click', async () => {
    stop();
    try {
      const response = await fetch('/api/reset', {method: 'POST'});
      showStatus(response.ok);
    } catch (_) { showStatus(false); }
  });
  window.addEventListener('blur', stop);
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  fetch('/api/state').then(r => r.json()).then(state => {
    limits = state.limits;
    controlsReady = true;
    $('limits').textContent = `范围：前后 ${limits.vx.toFixed(1)} m/s，横移 ${limits.vy.toFixed(1)} m/s，转向 ${limits.wz.toFixed(1)} rad/s`;
    showStatus(true);
    if (pressed.size > 0) beginKeyboardControl(); else send();
  }).catch(() => showStatus(false));
  requestAnimationFrame(updateKeyboard);
  setInterval(send, 20);
})();
</script>
</body>
</html>"""


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class DesktopJoystickWindow:
    """Own an isolated controller window process and its temporary browser profile."""

    def __init__(self, mode: str, process: subprocess.Popen | None = None, profile_dir: str | None = None):
        self.mode = mode
        self.process = process
        self.profile_dir = profile_dir

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self.profile_dir is not None:
            shutil.rmtree(self.profile_dir, ignore_errors=True)
            self.profile_dir = None

    def __str__(self) -> str:
        return self.mode


def open_joystick_window(url: str) -> DesktopJoystickWindow:
    """Open the controller in an isolated desktop app window."""
    for browser in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        executable = shutil.which(browser)
        if executable is not None:
            profile_dir = tempfile.mkdtemp(prefix="leggedlab_joystick_chrome_")
            try:
                process = subprocess.Popen(
                    [
                        executable,
                        f"--user-data-dir={profile_dir}",
                        f"--app={url}",
                        "--window-size=700,430",
                        "--no-first-run",
                        "--no-default-browser-check",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                shutil.rmtree(profile_dir, ignore_errors=True)
                continue
            return DesktopJoystickWindow("isolated desktop window", process, profile_dir)
    webbrowser.open(url, new=2)
    return DesktopJoystickWindow("default browser")


class VirtualJoystickServer:
    """Serve a local desktop UI and expose its latest bounded velocity command."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_vx: float = 2.0,
        max_vy: float = 1.0,
        max_wz: float = 2.0,
        timeout: float = 0.5,
    ) -> None:
        limits = (float(max_vx), float(max_vy), float(max_wz))
        if any(not math.isfinite(value) or value < 0.0 for value in limits):
            raise ValueError(f"joystick limits must be finite and non-negative, got {limits}")
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError(f"joystick timeout must be finite and positive, got {timeout}")
        self.host = host
        self.port = int(port)
        self.limits = limits
        self.timeout = float(timeout)
        self._command = (0.0, 0.0, 0.0)
        self._updated_at = 0.0
        self._reset_generation = 0
        self._consumed_reset_generation = 0
        self._lock = threading.Lock()
        self._httpd: _Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    self._reply(HTTPStatus.OK, _PAGE.encode(), "text/html; charset=utf-8")
                elif self.path == "/api/state":
                    self._reply_json(HTTPStatus.OK, owner.snapshot())
                else:
                    self._reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/api/reset":
                    owner.request_reset()
                    self._reply_json(HTTPStatus.OK, owner.snapshot())
                    return
                if self.path != "/api/command":
                    self._reply_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", "0")), 4096)
                    payload = json.loads(self.rfile.read(length))
                    owner.update(payload)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._reply_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._reply_json(HTTPStatus.OK, owner.snapshot())

            def _reply_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
                self._reply(status, json.dumps(payload).encode(), "application/json")

            def _reply(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._httpd = _Server((self.host, self.port), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="virtual-joystick", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        self._httpd = None
        self._thread = None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

    def update(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("command must be a JSON object")
        values = []
        for key, limit in zip(("vx", "vy", "wz"), self.limits, strict=True):
            value = float(payload.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            values.append(max(-limit, min(limit, value)))
        with self._lock:
            self._command = tuple(values)
            self._updated_at = time.monotonic()

    def command(self) -> tuple[float, float, float]:
        with self._lock:
            command = self._command
            age = time.monotonic() - self._updated_at
        return command if age <= self.timeout else (0.0, 0.0, 0.0)

    def request_reset(self) -> None:
        with self._lock:
            self._command = (0.0, 0.0, 0.0)
            self._updated_at = time.monotonic()
            self._reset_generation += 1

    def consume_reset(self) -> bool:
        with self._lock:
            if self._consumed_reset_generation == self._reset_generation:
                return False
            self._consumed_reset_generation = self._reset_generation
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            age = time.monotonic() - self._updated_at
        vx, vy, wz = self.command()
        return {
            "vx": vx,
            "vy": vy,
            "wz": wz,
            "active": age <= self.timeout,
            "limits": {"vx": self.limits[0], "vy": self.limits[1], "wz": self.limits[2]},
        }

    def __enter__(self) -> VirtualJoystickServer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
