"""浏览器版 LeggedLab 启动器。

这个脚本只依赖 Python 标准库，启动一个本地 HTTP 服务，用更现代的 Web UI
填写 train.py / play.py 参数并启动进程。它不替换原来的 Tkinter 启动器。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


@dataclass
class OptionSpec:
    """命令行参数定义。"""

    name: str
    takes_value: bool
    value_hint: str = ""
    default_value: str = ""
    group: str = "其他"
    choices: tuple[str, ...] = ()


class LauncherState:
    """负责参数发现、命令构造和子进程生命周期。"""

    CORE_OPTIONS = {"--task", "--load_run", "--checkpoint"}
    SCRIPT_NAMES = ("train", "play")
    EQUALS_VALUE_OPTIONS = {"--kit_args"}

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.scripts = {
            "train": self.repo_root / "legged_lab" / "scripts" / "train.py",
            "play": self.repo_root / "legged_lab" / "scripts" / "play.py",
        }
        self.python = sys.executable
        self.tasks = self._detect_tasks()
        self.options = {name: self._load_options(name) for name in self.SCRIPT_NAMES}
        self.logs: list[dict[str, Any]] = []
        self.log_seq = 0
        self.log_lock = threading.Lock()
        self.process_lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.process_pgid: int | None = None
        self.current_command: list[str] = []
        self.current_env_overrides: dict[str, str] = {}
        self.last_returncode: int | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.process_lock:
            running = self.process is not None and self.process.poll() is None
            pid = self.process.pid if running and self.process is not None else None
        return {
            "tasks": self.tasks,
            "options": {key: [asdict(item) for item in value] for key, value in self.options.items()},
            "checkpoints": self._list_checkpoints(limit=300),
            "latest_checkpoint": self._latest_checkpoint(),
            "python": self.python,
            "running": running,
            "pid": pid,
            "last_returncode": self.last_returncode,
            "command": self.current_command,
            "env": self.current_env_overrides,
        }

    def logs_since(self, since: int) -> dict[str, Any]:
        with self.log_lock:
            lines = [entry for entry in self.logs if int(entry["seq"]) > since]
            latest = self.log_seq
        with self.process_lock:
            running = self.process is not None and self.process.poll() is None
            pid = self.process.pid if running and self.process is not None else None
        return {
            "lines": lines,
            "latest": latest,
            "running": running,
            "pid": pid,
            "returncode": self.last_returncode,
            "command": self.current_command,
            "env": self.current_env_overrides,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        script_name = str(payload.get("script", "train"))
        if script_name not in self.scripts:
            raise ValueError(f"Unsupported script: {script_name}")

        with self.process_lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有任务在运行，请先停止。")

            command = self.build_command(payload)
            env_overrides = self._parse_env_overrides(str(payload.get("env", "")))
            env = os.environ.copy()
            env.update(env_overrides)

            self.last_returncode = None
            self.current_command = command
            self.current_env_overrides = env_overrides
            self._append_log("[INFO] 启动命令: " + self._format_shell_command(command, env_overrides) + "\n")

            popen_kwargs: dict[str, Any] = {
                "args": command,
                "cwd": str(self.repo_root),
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "env": env,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            self.process = subprocess.Popen(**popen_kwargs)
            self.process_pgid = None
            if os.name != "nt":
                try:
                    self.process_pgid = os.getpgid(self.process.pid)
                except Exception:
                    self.process_pgid = None

            reader = threading.Thread(target=self._read_process_output, args=(self.process,), daemon=True)
            reader.start()

        return {"ok": True, "pid": self.process.pid, "command": command, "env": env_overrides}

    def stop(self) -> dict[str, Any]:
        with self.process_lock:
            proc = self.process
            pgid = self.process_pgid
        if proc is None or proc.poll() is not None:
            self._append_log("[INFO] 当前无运行中的进程。\n")
            return {"ok": True, "stopped": False}

        self._append_log("[INFO] 正在停止进程...\n")
        self._terminate_process(proc, pgid)
        with self.process_lock:
            if self.process is proc:
                self.process = None
                self.process_pgid = None
        self._append_log("[INFO] 已发送停止信号。\n")
        return {"ok": True, "stopped": True}

    def build_command(self, payload: dict[str, Any]) -> list[str]:
        script_name = str(payload.get("script", "train"))
        if script_name not in self.scripts:
            raise ValueError(f"Unsupported script: {script_name}")

        core = payload.get("core") or {}
        options = payload.get("options") or {}
        extra = str(payload.get("extra", "")).strip()

        command = [self.python, "-u", str(self.scripts[script_name])]

        task_value = str(core.get("task", "")).strip()
        load_run_value = str(core.get("load_run", "")).strip()
        checkpoint_raw = str(core.get("checkpoint", "")).strip()
        checkpoint_value, run_from_path, task_from_path = self._resolve_checkpoint_input(checkpoint_raw)

        if run_from_path:
            load_run_value = run_from_path
        if task_from_path and not task_value:
            task_value = task_from_path

        if task_value:
            command.append(f"--task={task_value}")
        if checkpoint_value:
            if script_name == "train":
                command.append("--resume=True")
            if load_run_value:
                command.append(f"--load_run={load_run_value}")
            command.append(f"--checkpoint={checkpoint_value}")

        known_options = {item.name: item for item in self.options.get(script_name, [])}
        for name, value in options.items():
            if name in self.CORE_OPTIONS:
                continue
            spec = known_options.get(name)
            if spec is None:
                continue
            if spec.takes_value:
                text = str(value).strip()
                if not text:
                    continue
                self._append_value_arg(command, name, text)
            else:
                if bool(value):
                    command.append(name)

        if extra:
            command.extend(shlex.split(extra))
        return command

    def _append_value_arg(self, command: list[str], name: str, value: str) -> None:
        if name in self.EQUALS_VALUE_OPTIONS:
            command.append(f"{name}={value}")
            return
        try:
            pieces = shlex.split(value)
        except ValueError:
            pieces = [value]
        if len(pieces) > 1:
            command.append(name)
            command.extend(pieces)
        else:
            command.append(f"{name}={value}")

    def _read_process_output(self, proc: subprocess.Popen) -> None:
        if proc.stdout is not None:
            for line in proc.stdout:
                self._append_log(line)
        returncode = proc.wait()
        with self.process_lock:
            self.last_returncode = returncode
            if self.process is proc:
                self.process = None
                self.process_pgid = None
        self._append_log(f"\n[INFO] 进程结束，返回码: {returncode}\n")

    def _terminate_process(self, proc: subprocess.Popen, pgid: int | None) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            elif pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            self._append_log("[WARN] 进程未及时退出，正在强制停止...\n")
        except ProcessLookupError:
            return
        except Exception as exc:
            self._append_log(f"[WARN] 停止进程异常: {exc}\n")

        try:
            if os.name == "nt":
                proc.kill()
            elif pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception as exc:
            self._append_log(f"[WARN] 强制停止失败: {exc}\n")

    def _append_log(self, text: str) -> None:
        with self.log_lock:
            for line in text.splitlines(keepends=True):
                self.log_seq += 1
                self.logs.append({"seq": self.log_seq, "text": line, "time": time.time()})
            if len(self.logs) > 6000:
                self.logs = self.logs[-5000:]

    def _detect_tasks(self) -> list[str]:
        env_init = self.repo_root / "legged_lab" / "envs" / "__init__.py"
        if not env_init.exists():
            return []
        try:
            text = env_init.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            return []

        tasks: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "register"
                and isinstance(func.value, ast.Name)
                and func.value.id == "task_registry"
            ):
                continue
            if not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                tasks.append(first_arg.value)

        if not tasks:
            tasks = re.findall(r'task_registry\.register\(\s*"([a-zA-Z0-9_]+)"', text)

        ordered: list[str] = []
        seen: set[str] = set()
        for task in tasks:
            if task not in seen:
                ordered.append(task)
                seen.add(task)
        return ordered

    def _load_options(self, script_name: str) -> list[OptionSpec]:
        script_path = self.scripts[script_name]
        defaults = self._extract_default_values(script_path)
        source_specs = self._extract_options_from_source(script_path)
        help_specs = self._extract_options_from_help(script_path)
        specs = self._merge_option_specs(help_specs, source_specs)
        for spec in specs:
            spec.default_value = defaults.get(spec.name, "")
            spec.group = self._group_option(spec.name)
        return specs

    def _merge_option_specs(self, help_specs: list[OptionSpec], source_specs: list[OptionSpec]) -> list[OptionSpec]:
        source_by_name = {spec.name: spec for spec in source_specs}
        merged: list[OptionSpec] = []
        seen: set[str] = set()

        for help_spec in help_specs:
            source_spec = source_by_name.get(help_spec.name)
            if source_spec is not None:
                help_spec.takes_value = source_spec.takes_value
            merged.append(help_spec)
            seen.add(help_spec.name)

        for source_spec in source_specs:
            if source_spec.name not in seen:
                merged.append(source_spec)
                seen.add(source_spec.name)

        return merged or source_specs or help_specs

    def _extract_options_from_help(self, script_path: Path) -> list[OptionSpec]:
        cmd = [self.python, str(script_path), "--help"]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=20,
            )
        except Exception:
            return []
        if result.returncode not in (0, 1):
            return []

        specs: list[OptionSpec] = []
        seen: set[str] = set()
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            tokens = stripped.split()
            long_token_index = next((idx for idx, token in enumerate(tokens) if token.startswith("--")), None)
            if long_token_index is None:
                continue

            raw_option = tokens[long_token_index].rstrip(",")
            if "=" in raw_option:
                option_name, inline_hint = raw_option.split("=", 1)
                hint_tokens = [inline_hint] if inline_hint else []
            else:
                option_name = raw_option
                hint_tokens = self._extract_help_value_hint_tokens(tokens[long_token_index + 1 :])
            if option_name in {"--help", "-h"} or option_name in seen:
                continue

            value_hint = " ".join(hint_tokens).strip()
            choices = self._parse_choices(value_hint)
            specs.append(
                OptionSpec(
                    name=option_name,
                    takes_value=bool(value_hint),
                    value_hint=value_hint,
                    choices=choices,
                )
            )
            seen.add(option_name)
        return specs

    def _extract_help_value_hint_tokens(self, tokens: list[str]) -> list[str]:
        hint_tokens: list[str] = []
        for token in tokens:
            cleaned = token.rstrip(",")
            if cleaned.startswith("-"):
                break
            # argparse 的值占位符一般是大写或 choices: {a,b,c}。
            # 后续普通英文 help 说明不纳入 value_hint，避免把 flag 误判成带值参数。
            if cleaned.startswith("{") or cleaned.upper() == cleaned:
                hint_tokens.append(cleaned)
                continue
            break
        return hint_tokens

    def _parse_choices(self, value_hint: str) -> tuple[str, ...]:
        text = value_hint.strip()
        if not (text.startswith("{") and "}" in text):
            return ()
        body = text[1 : text.index("}")]
        choices = tuple(item.strip() for item in body.split(",") if item.strip())
        return choices

    def _extract_options_from_source(self, script_path: Path) -> list[OptionSpec]:
        specs: list[OptionSpec] = []
        seen: set[str] = set()
        for path in [script_path, self.repo_root / "legged_lab" / "utils" / "cli_args.py"]:
            if not path.exists():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn_name = ""
                if isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                if fn_name != "add_argument":
                    continue

                option_name = None
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        option_name = arg.value
                        break
                if not option_name or option_name in seen or option_name == "--help":
                    continue

                takes_value = self._source_arg_takes_value(node)
                choices = self._source_arg_choices(node)
                specs.append(OptionSpec(name=option_name, takes_value=takes_value, choices=choices))
                seen.add(option_name)
        return specs

    def _source_arg_takes_value(self, node: ast.Call) -> bool:
        action_node = next((kw.value for kw in node.keywords if kw.arg == "action"), None)
        action = action_node.value if isinstance(action_node, ast.Constant) and isinstance(action_node.value, str) else None
        if action in {"store_true", "store_false", "store_const", "append_const", "count", "help", "version"}:
            return False
        return True

    def _source_arg_choices(self, node: ast.Call) -> tuple[str, ...]:
        choices_node = next((kw.value for kw in node.keywords if kw.arg == "choices"), None)
        if choices_node is None:
            return ()
        if isinstance(choices_node, (ast.Set, ast.List, ast.Tuple)):
            values = []
            for item in choices_node.elts:
                if isinstance(item, ast.Constant):
                    values.append(str(item.value))
            return tuple(values)
        return ()

    def _extract_default_values(self, script_path: Path) -> dict[str, str]:
        defaults: dict[str, str] = {}
        paths = [script_path, self.repo_root / "legged_lab" / "utils" / "cli_args.py"]
        for path in paths:
            if not path.exists():
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            constants = self._collect_module_constants(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn_name = ""
                if isinstance(node.func, ast.Attribute):
                    fn_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    fn_name = node.func.id
                if fn_name != "add_argument":
                    continue
                option_name = None
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        option_name = arg.value
                        break
                if not option_name:
                    continue
                default_node = next((kw.value for kw in node.keywords if kw.arg == "default"), None)
                if default_node is None:
                    continue
                defaults[option_name] = self._default_to_text(self._eval_ast_default(default_node, constants))
        return defaults

    def _collect_module_constants(self, tree: ast.AST) -> dict[str, object]:
        constants: dict[str, object] = {}
        body = tree.body if isinstance(tree, ast.Module) else []
        for node in body:
            if isinstance(node, ast.Assign):
                value = self._eval_ast_default(node.value, constants)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                constants[node.target.id] = self._eval_ast_default(node.value, constants)
        return constants

    def _eval_ast_default(self, node: ast.AST, constants: dict[str, object]) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return constants.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self._eval_ast_default(node.operand, constants)
            if isinstance(value, (int, float)):
                return -value
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [self._eval_ast_default(item, constants) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and len(node.args) >= 2:
                return self._eval_ast_default(node.args[1], constants)
        return None

    def _default_to_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _group_option(self, name: str) -> str:
        if name in {"--task", "--num_envs", "--seed", "--runner", "--device", "--headless", "--enable_cameras"}:
            return "核心"
        if name.startswith("--wmp_") or name.startswith("--amp_"):
            return "WMP / AMP"
        if name in {
            "--max_iterations",
            "--num_steps_per_env",
            "--num_mini_batches",
            "--num_mini_batces",
            "--distributed",
        }:
            return "训练尺度"
        if name in {
            "--experiment_name",
            "--run_name",
            "--resume",
            "--load_run",
            "--checkpoint",
            "--logger",
            "--log_project_name",
            "--wandb_entity",
            "--wandb_mode",
            "--wandb_api_key",
        }:
            return "日志 / Checkpoint"
        if (
            name.startswith("--depth_point_")
            or name.startswith("--depth_image_")
            or name.startswith("--camera_")
            or name in {"--show_depth_points", "--show_depth_image", "--show_camera_axes", "--show_camera_model"}
        ):
            return "深度传感器"
        if name in {
            "--show_depth_points",
            "--show_height_scan_points",
            "--play_flat",
            "--play_render_interval",
            "--enable_play_push",
            "--hide_command",
        }:
            return "播放 / 传感器"
        if name in {"--livestream", "--xr", "--verbose", "--info", "--experience", "--rendering_mode", "--kit_args"}:
            return "IsaacSim"
        return "其他"

    def _list_checkpoints(self, limit: int = 300) -> list[dict[str, str]]:
        logs_dir = self.repo_root / "logs"
        if not logs_dir.exists():
            return []
        items: list[tuple[float, Path]] = []
        for path in logs_dir.glob("*/*/*.pt"):
            try:
                items.append((path.stat().st_mtime, path))
            except OSError:
                continue
        items.sort(reverse=True, key=lambda item: item[0])
        checkpoints: list[dict[str, str]] = []
        for _, path in items[:limit]:
            try:
                rel = path.resolve().relative_to(logs_dir.resolve())
            except Exception:
                continue
            if len(rel.parts) < 3:
                continue
            checkpoints.append(
                {
                    "task": rel.parts[0],
                    "run": rel.parts[1],
                    "checkpoint": path.name,
                    "path": str(path),
                    "label": f"{rel.parts[0]} / {rel.parts[1]} / {path.name}",
                }
            )
        return checkpoints

    def _latest_checkpoint(self) -> dict[str, str] | None:
        checkpoints = self._list_checkpoints(limit=1)
        return checkpoints[0] if checkpoints else None

    def _resolve_checkpoint_input(self, raw_value: str) -> tuple[str, str | None, str | None]:
        text = raw_value.strip()
        if not text:
            return "", None, None
        is_path_like = os.sep in text or (os.altsep and os.altsep in text)
        if not is_path_like:
            return text, None, None

        ckpt_path = Path(text)
        ckpt_name = ckpt_path.name
        if not ckpt_name:
            return text, None, None
        try:
            rel = ckpt_path.resolve().relative_to((self.repo_root / "logs").resolve())
        except Exception:
            return ckpt_name, None, None
        if len(rel.parts) >= 3:
            return ckpt_name, rel.parts[1], rel.parts[0]
        return ckpt_name, None, None

    def _parse_env_overrides(self, text: str) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if not text.strip():
            return overrides
        try:
            pieces = shlex.split(text)
        except ValueError:
            pieces = text.split()
        for piece in pieces:
            if "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            key = key.strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                overrides[key] = value
        return overrides

    def _format_shell_command(self, command: list[str], env_overrides: dict[str, str]) -> str:
        prefix = [f"{key}={shlex.quote(value)}" for key, value in env_overrides.items()]
        body = [shlex.quote(item) for item in command]
        return " ".join(prefix + body)


class LauncherRequestHandler(BaseHTTPRequestHandler):
    """HTTP API 与静态页面。"""

    server_version = "LeggedLabLauncher/1.0"

    @property
    def launcher(self) -> LauncherState:
        return self.server.launcher_state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # 保持终端干净，运行日志由页面显示。
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/state":
            self._send_json(self.launcher.snapshot())
            return
        if parsed.path == "/api/logs":
            query = urllib.parse.parse_qs(parsed.query)
            since = int(query.get("since", ["0"])[0] or 0)
            self._send_json(self.launcher.logs_since(since))
            return
        self._send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            payload = self._read_json()
            if parsed.path == "/api/start":
                self._send_json(self.launcher.start(payload))
                return
            if parsed.path == "/api/stop":
                self._send_json(self.launcher.stop())
                return
            if parsed.path == "/api/preview":
                command = self.launcher.build_command(payload)
                env_overrides = self.launcher._parse_env_overrides(str(payload.get("env", "")))
                self._send_json(
                    {
                        "ok": True,
                        "command": command,
                        "shell": self.launcher._format_shell_command(command, env_overrides),
                    }
                )
                return
            self._send_json({"ok": False, "error": "not found"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _find_free_port(host: str, requested_port: int) -> int:
    if requested_port != 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, requested_port)) != 0:
                return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="LeggedLab browser launcher.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port. Use 0 for a random free port.")
    parser.add_argument("--open", action="store_true", help="Open the launcher in the default browser.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    host = args.host
    port = _find_free_port(host, args.port)
    server = ThreadingHTTPServer((host, port), LauncherRequestHandler)
    server.launcher_state = LauncherState(repo_root)  # type: ignore[attr-defined]
    url = f"http://{host}:{port}"
    print(f"[INFO] LeggedLab Web Launcher: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down launcher.", flush=True)
    finally:
        try:
            server.launcher_state.stop()  # type: ignore[attr-defined]
        except Exception:
            pass
        server.server_close()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LeggedLab Launcher</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #eef1f5;
      --panel: #ffffff;
      --panel-2: #f8fafc;
      --text: #17202a;
      --muted: #667085;
      --line: #d9e0e8;
      --blue: #2557a7;
      --green: #1f7a4d;
      --red: #b42318;
      --amber: #9a6700;
      --focus: rgba(37, 87, 167, 0.20);
      --radius: 8px;
      font-family: Ubuntu, Inter, "Segoe UI", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      padding: 9px 12px;
      cursor: pointer;
      transition: border-color .15s, background .15s, box-shadow .15s, transform .05s;
    }
    button:hover { border-color: #9eb3d4; background: #f7fbff; }
    button:active { transform: translateY(1px); }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: white;
    }
    button.stop {
      background: #fff5f5;
      border-color: #f1b8b8;
      color: var(--red);
    }
    button.ghost {
      background: transparent;
    }
    button.env-chip {
      text-align: left;
      background: #fff;
      color: var(--text);
      border-color: var(--line);
      display: grid;
      gap: 3px;
      position: relative;
    }
    button.env-chip.active {
      background: #e8f2ff;
      color: #174987;
      border-color: #7aa7e8;
      box-shadow: 0 0 0 3px rgba(37, 87, 167, .12);
    }
    button.env-chip.active::after {
      content: "已启用";
      position: absolute;
      right: 10px;
      top: 9px;
      border-radius: 999px;
      background: #2557a7;
      color: #fff;
      font-size: 11px;
      font-weight: 800;
      padding: 2px 7px;
    }
    button.env-chip span:first-child {
      font-weight: 800;
      padding-right: 54px;
    }
    button.env-chip span:last-child {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      padding: 9px 10px;
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: #6b8fd2;
      box-shadow: 0 0 0 3px var(--focus);
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: .02em;
      text-transform: uppercase;
    }
    .shell {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.86);
      backdrop-filter: blur(10px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background:
        linear-gradient(135deg, #2557a7 0 52%, #1f7a4d 52% 100%);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.35);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #98a2b3;
    }
    .dot.running { background: var(--green); box-shadow: 0 0 0 4px rgba(31,122,77,.14); }
    main {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(380px, 420px) minmax(420px, 1fr) 460px;
      gap: 12px;
      padding: 12px;
    }
    section, aside {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .left, .middle, .right {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    .panel-head {
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      background: var(--panel-2);
    }
    .panel-title {
      margin: 0;
      font-size: 14px;
      font-weight: 800;
    }
    .panel-body {
      min-height: 0;
      overflow: auto;
      padding: 14px;
    }
    .left .panel-body {
      overflow-y: auto;
      overflow-x: hidden;
      padding-bottom: 22px;
    }
    .stack { display: grid; gap: 12px; }
    .segmented {
      display: grid;
      grid-template-columns: 1fr 1fr;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f3f6fa;
      gap: 4px;
    }
    .segmented button {
      border: 0;
      background: transparent;
      padding: 8px;
      font-weight: 800;
    }
    .segmented button.active {
      background: #fff;
      color: var(--blue);
      box-shadow: 0 1px 2px rgba(16,24,40,.10);
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .preset-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .preset-grid button {
      text-align: left;
      display: grid;
      grid-template-columns: 1fr;
      align-items: start;
      gap: 10px;
      min-height: 44px;
    }
    .preset-grid span {
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .preset-grid span:last-child { color: var(--muted); }
    .option-toolbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      width: 100%;
    }
    .option-groups {
      display: grid;
      gap: 12px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    summary {
      padding: 11px 12px;
      cursor: pointer;
      font-weight: 800;
      background: #f9fafb;
      border-bottom: 1px solid var(--line);
      list-style: none;
    }
    summary::-webkit-details-marker { display: none; }
    .option-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
      padding: 12px;
    }
    .option-row {
      border: 1px solid #e4e9f0;
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      display: grid;
      gap: 7px;
      position: relative;
    }
    .option-row:hover {
      border-color: #a8bddc;
      box-shadow: 0 2px 10px rgba(16, 24, 40, .06);
    }
    .option-row.hidden { display: none; }
    details.hidden { display: none; }
    .option-row:hover::after {
      display: none;
    }
    .option-row:hover::before {
      display: none;
    }
    .option-name {
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .name-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .unit-badge {
      flex: 0 0 auto;
      border: 1px solid #c8d7eb;
      background: #f3f8ff;
      color: #2557a7;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 800;
      text-transform: none;
      letter-spacing: 0;
    }
    .check-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 36px;
    }
    .check-row input {
      width: 18px;
      height: 18px;
    }
    .hint {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .right {
      grid-template-rows: auto auto 1fr;
    }
    .command-box {
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111827;
      color: #d1fae5;
      font-family: "Ubuntu Mono", "DejaVu Sans Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      max-height: 150px;
      overflow: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .control-row {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 8px;
    }
    .log-pane {
      margin: 0;
      height: 100%;
      min-height: 240px;
      overflow: auto;
      padding: 12px;
      background: #0b1020;
      color: #d7dde8;
      font-family: "Ubuntu Mono", "DejaVu Sans Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .empty {
      color: var(--muted);
      padding: 22px;
      text-align: center;
    }
    .floating-tooltip {
      position: fixed;
      z-index: 2147483647;
      max-width: min(420px, calc(100vw - 24px));
      background: #101828;
      color: #fff;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 12px;
      line-height: 1.5;
      box-shadow: 0 14px 32px rgba(16, 24, 40, .28);
      pointer-events: none;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity .08s, transform .08s;
      white-space: normal;
    }
    .floating-tooltip.visible {
      opacity: 1;
      transform: translateY(0);
    }
    .env-chip.play-hidden {
      display: none;
    }
    @media (max-width: 1180px) {
      main { grid-template-columns: minmax(340px, 380px) 1fr; }
      .right { grid-column: 1 / -1; min-height: 420px; }
    }
    @media (max-width: 760px) {
      .shell { height: auto; min-height: 100vh; }
      main { grid-template-columns: 1fr; }
      section, aside { min-height: 420px; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true"></div>
        <div>
          <h1>LeggedLab Launcher</h1>
          <div class="hint" id="pythonPath"></div>
        </div>
      </div>
      <div class="status">
        <span class="pill"><span id="runDot" class="dot"></span><span id="runState">未运行</span></span>
        <span class="pill" id="pidState">PID -</span>
      </div>
    </header>
    <main>
      <section class="left">
        <div class="panel-head"><h2 class="panel-title">运行</h2></div>
        <div class="panel-body stack">
          <div class="segmented">
            <button id="scriptTrain" type="button">Train</button>
            <button id="scriptPlay" type="button">Play</button>
          </div>
          <label>Task
            <select id="taskSelect"></select>
          </label>
          <label>Checkpoint
            <select id="checkpointSelect"></select>
          </label>
          <label>环境变量
            <input id="envInput" placeholder="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" />
          </label>
          <div class="preset-grid" id="envChips">
            <button type="button" class="env-chip" data-env="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True">
              <span>CUDA 内存碎片优化</span><span>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</span>
            </button>
            <button type="button" class="env-chip" data-env="WANDB_MODE=offline" data-env-key="WANDB_MODE">
              <span>WandB 离线</span><span>WANDB_MODE=offline</span>
            </button>
            <button type="button" class="env-chip" data-env="WANDB_MODE=disabled" data-env-key="WANDB_MODE">
              <span>关闭 WandB</span><span>WANDB_MODE=disabled</span>
            </button>
          </div>
          <div class="preset-grid" id="presetGrid"></div>
        </div>
      </section>

      <section class="middle">
        <div class="panel-head">
          <h2 class="panel-title">参数</h2>
          <div class="option-toolbar">
            <input id="filterInput" placeholder="过滤参数，例如 wmp / camera / logger" />
            <button id="resetDefaults" type="button">重置</button>
          </div>
        </div>
        <div class="panel-body">
          <div id="optionGroups" class="option-groups"></div>
        </div>
      </section>

      <aside class="right">
        <div class="panel-head"><h2 class="panel-title">命令</h2></div>
        <div class="panel-body stack">
          <pre id="commandPreview" class="command-box"></pre>
          <label>额外参数
            <textarea id="extraInput" rows="3" placeholder="这里会原样按 shell 规则拼接到命令末尾"></textarea>
          </label>
          <div class="control-row">
            <button id="startBtn" class="primary" type="button">启动</button>
            <button id="stopBtn" class="stop" type="button">停止</button>
            <button id="clearLogBtn" class="ghost" type="button">清屏</button>
          </div>
        </div>
        <div class="panel-head"><h2 class="panel-title">日志</h2></div>
        <pre id="logPane" class="log-pane"></pre>
      </aside>
    </main>
    <div id="floatingTooltip" class="floating-tooltip" role="tooltip"></div>
  </div>
  <script>
    const app = {
      data: null,
      script: "train",
      values: { train: {}, play: {} },
      core: { task: "", load_run: "", checkpoint: "" },
      env: "",
      extra: "",
      latestLog: 0,
      previewTimer: null,
      currentTrainPreset: "a1Full",
    };

    const $ = (id) => document.getElementById(id);

    const PARAM_HELP = {
      "--task": "任务名称。决定使用哪个机器人、地形、奖励和训练配置。",
      "--num_envs": "并行环境数量。训练越大吞吐越高，但显存、CPU 内存和相机开销也更大。",
      "--seed": "随机种子。传 -1 时由脚本随机生成。",
      "--runner": "Runner 类型。A1/B2 的 WMP-AMP-PPO 训练和播放应选择 wmp_amp。",
      "--max_iterations": "最大训练迭代数。resume 时表示训练到该目标迭代为止。",
      "--num_steps_per_env": "每个环境一次 rollout 采集的 policy step 数，影响 batch 大小和更新频率。",
      "--num_mini_batches": "PPO 每轮更新拆分的 mini-batch 数量。",
      "--experiment_name": "日志根目录名，最终路径是 logs/{experiment_name}/...",
      "--run_name": "本次 run 的名字后缀，会追加到时间戳目录后面。",
      "--resume": "是否从 checkpoint 恢复训练。常用写法是 --resume=True。",
      "--load_run": "要加载的 run 文件夹名，通常由 checkpoint 下拉自动填充。",
      "--checkpoint": "要加载的模型文件名，例如 model_10000.pt。",
      "--logger": "日志后端。训练推荐 wandb；快速测试可用 tensorboard。",
      "--log_project_name": "wandb/neptune 项目名。",
      "--wandb_entity": "wandb 用户或团队名。",
      "--wandb_mode": "wandb 模式：online 在线、offline 离线、disabled 关闭。",
      "--wandb_api_key": "wandb API key。通常从环境变量 WANDB_API_KEY 读取。",
      "--distributed": "多 GPU / 多节点训练开关。",
      "--amp_reward_coef": "AMP 判别器奖励系数，控制模仿奖励在总奖励中的权重。",
      "--amp_task_reward_lerp": "AMP 奖励与任务奖励混合系数。数值越大越偏向 task reward。",
      "--wmp_camera_num_envs": "真实 WMP 深度相机环境数量。可小于 num_envs，用部分相机降低渲染成本。",
      "--wmp_depth_training_iters": "DepthPredictor 每次触发时训练多少个梯度迭代。",
      "--wmp_depth_batch_size": "DepthPredictor 训练 batch size。",
      "--wmp_train_steps_per_iter": "每个 PPO iteration 内 world model 训练的梯度步数。",
      "--wmp_train_interval": "world model 每隔多少个 PPO iteration 训练一次。",
      "--headless": "无 GUI 运行。正式训练建议开启。",
      "--livestream": "IsaacSim livestream 模式。一般本地训练不需要。",
      "--enable_cameras": "启用相机传感器和 RTX 依赖。完整 WMP depth 训练必须开启。",
      "--xr": "启用 VR/AR XR 模式。",
      "--device": "仿真设备，例如 cuda、cuda:0 或 cpu。",
      "--verbose": "开启 SimulationApp verbose 日志。",
      "--info": "开启 SimulationApp info 日志。",
      "--experience": "指定 IsaacSim experience 文件。通常不需要手动设置。",
      "--rendering_mode": "渲染质量模式。训练建议 performance，画质检查可用 balanced/quality。",
      "--kit_args": "额外传给 Omniverse Kit 的参数字符串。",
      "--anim_recording_enabled": "启用 USD animation 记录。",
      "--anim_recording_start_time": "USD animation 开始记录时间。",
      "--anim_recording_stop_time": "USD animation 停止记录时间。",
      "--play_flat": "播放时替换为平地，同时保留 WMP sensor 观测维度，用来验证基础步态。",
      "--play_render_interval": "播放渲染间隔。数值越小越流畅但越卡；数值越大越省性能。",
      "--show_depth_image": "显示或保存 WMP 64x64 深度图。会自动启用相机。",
      "--show_rssm_depth_compare": "并排显示真实深度图和 RSSM prior 预测深度图，并打印 MSE/MAE。",
      "--depth_image_mode": "深度图输出模式：auto 优先 IsaacSim 内置窗口，kit 只用内置窗口，save 只保存 PNG，window 只尝试 OpenCV 窗口。",
      "--depth_image_dir": "深度图 PNG 保存目录。",
      "--depth_image_save_interval": "每隔多少个 play step 保存一张深度图。",
      "--show_depth_points": "把深度图反投影为红色 debug 点，用于检查相机看到的地面/障碍物。",
      "--show_camera_axes": "显示相机局部坐标轴，不会被 depth 相机渲染到图像里。",
      "--camera_axis_length": "相机坐标轴长度。",
      "--camera_axis_width": "相机坐标轴线宽。",
      "--show_camera_model": "在机器人相机位置显示一个小相机外壳，用于确认安装位置和朝向。",
      "--show_height_scan_points": "显示 height scanner 的 ray hit 点。",
      "--enable_play_push": "播放时保留随机推扰。默认播放会关闭 push，方便观察策略本身。",
      "--hide_command": "隐藏命令速度/当前速度可视化。",
      "--depth_point_stride": "深度点云采样步长。值越小点越密、渲染越重。",
      "--depth_point_max": "最多绘制多少个深度点。",
      "--depth_point_size": "深度点显示尺寸。",
      "--depth_point_forward_min": "只显示距离相机大于该值的深度点，用来滤掉近处遮挡。",
      "--depth_point_forward_max": "只显示距离相机小于该值的深度点。",
      "--depth_point_min_z": "只显示世界坐标 z 高于该值的深度点。",
      "--depth_point_max_z": "只显示世界坐标 z 低于该值的深度点。",
      "--depth_point_debug": "定期打印 depth 点统计信息。",
      "--depth_point_lift": "绘制点时沿世界 z 方向抬高，避免点被地面遮住。",
      "--depth_point_draw_rays": "从相机到点绘制黄色射线。仅保留给手写命令调试，GUI 默认隐藏。",
      "--depth_point_camera_index": "选择第几个 depth camera 可视化。-1 表示显示所有相机环境。",
      "--camera_offset_pos": "播放时覆盖相机相对机器人挂载位置，顺序为 x y z。",
      "--camera_offset_rot": "播放时覆盖相机相对旋转四元数，顺序为 w x y z。",
      "--camera_random_pitch_deg": "播放时覆盖相机 pitch 随机范围。",
      "--camera_fov_deg": "播放时覆盖相机水平视场角。",
      "--camera_disable_random_rotation": "播放时关闭相机随机旋转。用于固定相机视角排查 depth。"
    };

    const PARAM_UNITS = {
      "--play_render_interval": "sim step",
      "--depth_image_save_interval": "policy step",
      "--depth_point_stride": "pixel",
      "--depth_point_max": "points",
      "--depth_point_size": "px",
      "--depth_point_forward_min": "m",
      "--depth_point_forward_max": "m",
      "--depth_point_min_z": "m",
      "--depth_point_max_z": "m",
      "--depth_point_lift": "m",
      "--depth_point_camera_index": "index",
      "--camera_axis_length": "m",
      "--camera_axis_width": "px",
      "--camera_offset_pos": "m, m, m",
      "--camera_offset_rot": "wxyz",
      "--camera_random_pitch_deg": "deg",
      "--camera_fov_deg": "deg",
      "--wmp_depth_batch_size": "samples",
      "--wmp_depth_training_iters": "grad steps",
      "--wmp_train_steps_per_iter": "grad steps",
      "--wmp_train_interval": "PPO iter",
      "--wmp_camera_num_envs": "envs",
      "--num_envs": "envs",
      "--max_iterations": "iter",
      "--num_steps_per_env": "steps/env",
      "--num_mini_batches": "batches",
      "--anim_recording_start_time": "s",
      "--anim_recording_stop_time": "s"
    };

    const TRAIN_DEFAULTS = {
      "--num_envs": "4096",
      "--seed": "",
      "--runner": "wmp_amp",
      "--max_iterations": "20000",
      "--num_steps_per_env": "24",
      "--num_mini_batches": "",
      "--experiment_name": "",
      "--run_name": "a1_wmp_amp_4096env_1024cam_original_v1",
      "--logger": "wandb",
      "--log_project_name": "a1_wmp_amp",
      "--wandb_entity": "",
      "--wandb_mode": "",
      "--wandb_api_key": "",
      "--distributed": false,
      "--wmp_camera_num_envs": "1024",
      "--wmp_depth_training_iters": "",
      "--wmp_depth_batch_size": "",
      "--wmp_train_steps_per_iter": "",
      "--wmp_train_interval": "",
      "--headless": true,
      "--livestream": "",
      "--enable_cameras": true,
      "--xr": false,
      "--device": "cuda:0",
      "--verbose": false,
      "--info": false,
      "--experience": "",
      "--rendering_mode": "performance",
      "--kit_args": "",
      "--anim_recording_enabled": false,
      "--anim_recording_start_time": "",
      "--anim_recording_stop_time": ""
    };

    const PLAY_DEFAULTS = {
      "--num_envs": "4",
      "--seed": "",
      "--runner": "wmp_amp",
      "--play_render_interval": "4",
      "--depth_image_mode": "auto",
      "--depth_image_save_interval": "10",
      "--camera_axis_length": "0.20",
      "--camera_axis_width": "3.0",
      "--depth_point_stride": "8",
      "--depth_point_max": "4096",
      "--depth_point_size": "1.0",
      "--depth_point_forward_min": "0.01",
      "--depth_point_forward_max": "5.0",
      "--depth_point_lift": "0.05",
      "--depth_point_camera_index": "0",
      "--camera_offset_pos": "",
      "--camera_offset_rot": "",
      "--camera_random_pitch_deg": "",
      "--camera_fov_deg": "",
      "--headless": false,
      "--livestream": "",
      "--enable_cameras": false,
      "--device": "cuda:0",
      "--rendering_mode": "balanced",
      "--kit_args": ""
    };

    function quoteShell(text) {
      if (!text) return "''";
      if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text;
      return "'" + text.replaceAll("'", "'\\''") + "'";
    }

    function paramHelp(spec) {
      const unit = PARAM_UNITS[spec.name] ? `单位：${PARAM_UNITS[spec.name]}。` : "";
      const fallback = spec.value_hint ? `参数值格式：${spec.value_hint}。` : "这是一个开关参数，勾选后会传入命令。";
      return `${PARAM_HELP[spec.name] || fallback}${unit ? " " + unit : ""}`;
    }

    function paramPlaceholder(spec) {
      const unit = PARAM_UNITS[spec.name];
      if (unit) return spec.value_hint ? `${spec.value_hint} · ${unit}` : unit;
      return spec.value_hint || "";
    }

    function optionSpecs() {
      return (app.data.options[app.script] || []).filter((spec) => isVisibleOption(spec.name));
    }

    function playAllowedOptionNames() {
      return new Set([
        "--num_envs",
        "--seed",
        "--runner",
        "--play_flat",
        "--play_render_interval",
        "--show_depth_image",
        "--show_rssm_depth_compare",
        "--depth_image_mode",
        "--depth_image_dir",
        "--depth_image_save_interval",
        "--show_depth_points",
        "--show_camera_axes",
        "--camera_axis_length",
        "--camera_axis_width",
        "--show_camera_model",
        "--show_height_scan_points",
        "--enable_play_push",
        "--hide_command",
        "--depth_point_stride",
        "--depth_point_max",
        "--depth_point_size",
        "--depth_point_forward_min",
        "--depth_point_forward_max",
        "--depth_point_min_z",
        "--depth_point_max_z",
        "--depth_point_debug",
        "--depth_point_lift",
        "--depth_point_camera_index",
        "--camera_offset_pos",
        "--camera_offset_rot",
        "--camera_random_pitch_deg",
        "--camera_fov_deg",
        "--camera_disable_random_rotation",
        "--headless",
        "--livestream",
        "--enable_cameras",
        "--device",
        "--rendering_mode",
        "--kit_args"
      ]);
    }

    function isVisibleOption(name) {
      if (hiddenOptionNames().has(name)) return false;
      if (app.script === "play") return playAllowedOptionNames().has(name);
      return true;
    }

    function hiddenOptionNames() {
      const hidden = new Set([
        "--resume",
        "--amp_num_preload_transitions",
        "--amp_reward_coef",
        "--amp_task_reward_lerp",
        "--depth_point_draw_rays"
      ]);
      return hidden;
    }

    function ensureDefaults(script) {
      const bucket = app.values[script];
      const previousScript = app.script;
      app.script = script;
      const uiDefaults = script === "train" ? TRAIN_DEFAULTS : PLAY_DEFAULTS;
      for (const spec of (app.data.options[script] || []).filter((item) => isVisibleOption(item.name))) {
        if (spec.name in bucket) continue;
        if (spec.name in uiDefaults) {
          bucket[spec.name] = uiDefaults[spec.name];
        } else if (spec.takes_value) {
          bucket[spec.name] = spec.default_value || "";
        } else {
          bucket[spec.name] = String(spec.default_value || "").toLowerCase() === "true";
        }
      }
      app.script = previousScript;
    }

    function setScript(script) {
      app.script = script;
      ensureDefaults(script);
      $("scriptTrain").classList.toggle("active", script === "train");
      $("scriptPlay").classList.toggle("active", script === "play");
      if (script === "play") removeEnvByKey("WANDB_MODE");
      renderPresets();
      renderOptions();
      syncCoreInputs();
      updatePreviewSoon();
    }

    function presetSpecs() {
      if (app.script === "play") {
        return [
          { name: "playFlat", title: "平地播放", subtitle: "验证步态" },
          { name: "depthImage", title: "保存 Depth 图", subtitle: "64x64 PNG" },
          { name: "depthPoints", title: "显示 Depth 点", subtitle: "所有相机" },
        ];
      }
      return [
        { name: "a1Full", title: "A1 WMP 正式训练", subtitle: "4096 / 1024 cam" },
        { name: "a1Smoke", title: "A1 WMP 快速训练", subtitle: "2 env" },
      ];
    }

    function renderPresets() {
      const grid = $("presetGrid");
      grid.innerHTML = "";
      if (app.script === "train") {
        const label = document.createElement("label");
        label.textContent = "训练预设";
        const select = document.createElement("select");
        select.id = "trainPresetSelect";
        for (const preset of presetSpecs()) {
          const option = document.createElement("option");
          option.value = preset.name;
          option.textContent = `${preset.title} · ${preset.subtitle}`;
          select.appendChild(option);
        }
        select.value = app.currentTrainPreset || "a1Full";
        select.addEventListener("change", () => {
          app.currentTrainPreset = select.value;
          applyPreset(select.value);
        });
        label.appendChild(select);
        grid.appendChild(label);
        return;
      }
      for (const preset of presetSpecs()) {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.preset = preset.name;
        const title = document.createElement("span");
        title.textContent = preset.title;
        const subtitle = document.createElement("span");
        subtitle.textContent = preset.subtitle;
        button.appendChild(title);
        button.appendChild(subtitle);
        button.addEventListener("click", () => applyPreset(preset.name));
        grid.appendChild(button);
      }
    }

    function renderTasks() {
      const select = $("taskSelect");
      select.innerHTML = "";
      for (const task of app.data.tasks) {
        const opt = document.createElement("option");
        opt.value = task;
        opt.textContent = task;
        select.appendChild(opt);
      }
      const preferred = app.data.tasks.includes("a1_wmp_amp_terrain") ? "a1_wmp_amp_terrain" : (app.data.tasks[0] || "");
      app.core.task = app.core.task || preferred;
      select.value = app.core.task;
    }

    function renderCheckpoints() {
      const select = $("checkpointSelect");
      select.innerHTML = "";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = "不选择 checkpoint";
      select.appendChild(empty);
      for (const ckpt of app.data.checkpoints) {
        const opt = document.createElement("option");
        opt.value = ckpt.path;
        opt.textContent = ckpt.label;
        select.appendChild(opt);
      }
      syncCoreInputs();
    }

    function syncCoreInputs() {
      $("taskSelect").value = app.core.task;
      const selectedPath = checkpointPathFromCore();
      $("checkpointSelect").value = selectedPath || "";
      $("envInput").value = app.env || "";
      $("extraInput").value = app.extra || "";
      syncEnvChips();
    }

    function checkpointPathFromCore() {
      if (!app.core.task || !app.core.load_run || !app.core.checkpoint) return "";
      const match = app.data.checkpoints.find((item) =>
        item.task === app.core.task && item.run === app.core.load_run && item.checkpoint === app.core.checkpoint
      );
      return match ? match.path : "";
    }

    function clearCheckpointSelection() {
      app.core.load_run = "";
      app.core.checkpoint = "";
    }

    function setCheckpointFromItem(ckpt) {
      if (!ckpt) {
        clearCheckpointSelection();
        return;
      }
      app.core.task = ckpt.task;
      app.core.load_run = ckpt.run;
      app.core.checkpoint = ckpt.checkpoint;
    }

    function groupOrder(name) {
      const order = [
        "核心",
        "训练尺度",
        "WMP / AMP",
        "日志 / Checkpoint",
        "播放 / 传感器",
        "深度传感器",
        "IsaacSim",
        "其他"
      ];
      const index = order.indexOf(name);
      return index < 0 ? 999 : index;
    }

    function renderOptions() {
      const root = $("optionGroups");
      root.innerHTML = "";
      const groups = new Map();
      for (const spec of optionSpecs()) {
        if (["--task", "--load_run", "--checkpoint"].includes(spec.name)) continue;
        if (!groups.has(spec.group)) groups.set(spec.group, []);
        groups.get(spec.group).push(spec);
      }
      const sorted = Array.from(groups.entries()).sort((a, b) => groupOrder(a[0]) - groupOrder(b[0]));
      for (const [group, specs] of sorted) {
        const details = document.createElement("details");
        details.open = true;
        details.dataset.group = group;
        const summary = document.createElement("summary");
        summary.textContent = `${group} · ${specs.length}`;
        details.appendChild(summary);
        const list = document.createElement("div");
        list.className = "option-list";
        for (const spec of specs) list.appendChild(renderOption(spec));
        details.appendChild(list);
        root.appendChild(details);
      }
      applyFilter();
    }

    function renderOption(spec) {
      const row = document.createElement("div");
      row.className = "option-row";
      row.dataset.group = spec.group;
      row.dataset.name = spec.name.toLowerCase();
      const helpText = paramHelp(spec);
      row.dataset.hint = `${spec.value_hint || ""} ${helpText}`.toLowerCase();
      row.dataset.help = helpText;
      attachTooltip(row, helpText);

      if (spec.takes_value) {
        row.appendChild(renderNameLine(spec));
        const input = renderValueControl(spec, helpText);
        row.appendChild(input);
        const hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = optionHintText(spec, helpText);
        row.appendChild(hint);
      } else {
        const wrap = document.createElement("div");
        wrap.className = "check-row";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = Boolean(app.values[app.script][spec.name]);
        input.removeAttribute("title");
        input.addEventListener("change", () => {
          app.values[app.script][spec.name] = input.checked;
          updatePreviewSoon();
        });
        wrap.appendChild(renderNameLine(spec));
        wrap.appendChild(input);
        row.appendChild(wrap);
      }
      return row;
    }

    function renderValueControl(spec, helpText) {
      if (spec.choices && spec.choices.length > 0) {
        const select = document.createElement("select");
        const empty = document.createElement("option");
        empty.value = "";
        empty.textContent = `使用默认${spec.default_value ? ` (${spec.default_value})` : ""}`;
        select.appendChild(empty);
        for (const choice of spec.choices) {
          const option = document.createElement("option");
          option.value = choice;
          option.textContent = choice;
          select.appendChild(option);
        }
        select.value = app.values[app.script][spec.name] ?? "";
        select.addEventListener("change", () => {
          app.values[app.script][spec.name] = select.value;
          updatePreviewSoon();
        });
        return select;
      }

      const input = document.createElement("input");
      input.value = app.values[app.script][spec.name] ?? spec.default_value ?? "";
      input.placeholder = paramPlaceholder(spec);
      input.removeAttribute("title");
      input.addEventListener("input", () => {
        app.values[app.script][spec.name] = input.value;
        updatePreviewSoon();
      });
      return input;
    }

    function optionHintText(spec, helpText) {
      if (spec.choices && spec.choices.length > 0) {
        return `可选：${spec.choices.join(" / ")}`;
      }
      return PARAM_UNITS[spec.name] ? `单位：${PARAM_UNITS[spec.name]}` : helpText;
    }

    function attachTooltip(element, text) {
      element.addEventListener("mouseenter", () => showTooltip(element, text));
      element.addEventListener("mousemove", () => positionTooltip(element));
      element.addEventListener("mouseleave", hideTooltip);
      element.addEventListener("focusin", () => showTooltip(element, text));
      element.addEventListener("focusout", hideTooltip);
    }

    function showTooltip(anchor, text) {
      const tooltip = $("floatingTooltip");
      tooltip.textContent = text;
      tooltip.classList.add("visible");
      positionTooltip(anchor);
    }

    function positionTooltip(anchor) {
      const tooltip = $("floatingTooltip");
      if (!tooltip.classList.contains("visible")) return;
      const rect = anchor.getBoundingClientRect();
      const gap = 10;
      const tooltipRect = tooltip.getBoundingClientRect();
      let left = rect.left;
      let top = rect.top - tooltipRect.height - gap;
      if (top < 8) top = rect.bottom + gap;
      if (left + tooltipRect.width > window.innerWidth - 8) {
        left = window.innerWidth - tooltipRect.width - 8;
      }
      left = Math.max(8, left);
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${Math.max(8, top)}px`;
    }

    function hideTooltip() {
      $("floatingTooltip").classList.remove("visible");
    }

    function renderNameLine(spec) {
      const line = document.createElement("div");
      line.className = "name-line";
      const name = document.createElement("div");
      name.className = "option-name";
      name.textContent = spec.name;
      line.appendChild(name);
      if (PARAM_UNITS[spec.name]) {
        const unit = document.createElement("span");
        unit.className = "unit-badge";
        unit.textContent = PARAM_UNITS[spec.name];
        line.appendChild(unit);
      }
      return line;
    }

    function collectPayload() {
      const specs = optionSpecs();
      const options = {};
      for (const spec of specs) {
        if (["--task", "--load_run", "--checkpoint"].includes(spec.name)) continue;
        const value = app.values[app.script][spec.name];
        if (spec.takes_value) {
          const text = String(value ?? "").trim();
          if (text && text !== String(spec.default_value ?? "")) options[spec.name] = text;
        } else if (value) {
          options[spec.name] = true;
        }
      }
      return {
        script: app.script,
        core: { ...app.core },
        options,
        env: app.env,
        extra: app.extra,
      };
    }

    function updatePreviewSoon() {
      clearTimeout(app.previewTimer);
      app.previewTimer = setTimeout(updatePreview, 120);
    }

    async function updatePreview() {
      const payload = collectPayload();
      try {
        const result = await postJson("/api/preview", payload);
        $("commandPreview").textContent = result.shell || "";
      } catch (err) {
        $("commandPreview").textContent = String(err);
      }
    }

    function applyFilter() {
      const keyword = $("filterInput").value.trim().toLowerCase();
      for (const row of document.querySelectorAll(".option-row")) {
        const hit = !keyword || row.dataset.name.includes(keyword) || row.dataset.hint.includes(keyword);
        row.classList.toggle("hidden", !hit);
      }
    }

    function setValue(name, value, script = app.script) {
      ensureDefaults(script);
      app.values[script][name] = value;
    }

    function resetValuesForScript(script) {
      app.values[script] = {};
      ensureDefaults(script);
    }

    function envTokens() {
      const text = app.env.trim();
      if (!text) return [];
      try {
        return text.match(/(?:[^\s"']+|"[^"]*"|'[^']*')+/g) || [];
      } catch (err) {
        return text.split(/\s+/).filter(Boolean);
      }
    }

    function toggleEnvToken(token) {
      const tokenKey = envTokenKey(token);
      const tokens = envTokens().filter((item) => item !== token);
      const wasSelected = envTokens().includes(token);
      if (!wasSelected) {
        const withoutSameKey = tokens.filter((item) => envTokenKey(item) !== tokenKey);
        withoutSameKey.push(token);
        app.env = withoutSameKey.join(" ");
      } else {
        app.env = tokens.join(" ");
      }
      $("envInput").value = app.env;
      syncEnvChips();
      updatePreviewSoon();
    }

    function envTokenKey(token) {
      return String(token).split("=", 1)[0] || token;
    }

    function removeEnvByKey(key) {
      const tokens = envTokens().filter((item) => envTokenKey(item) !== key);
      app.env = tokens.join(" ");
      if ($("envInput")) $("envInput").value = app.env;
      syncEnvChips();
    }

    function setEnvText(text) {
      app.env = text;
      $("envInput").value = app.env;
      syncEnvChips();
      updatePreviewSoon();
    }

    function syncEnvChips() {
      const selected = new Set(envTokens());
      for (const chip of document.querySelectorAll("[data-env]")) {
        chip.classList.toggle("active", selected.has(chip.dataset.env));
        const hideInPlay = app.script === "play" && chip.dataset.envKey === "WANDB_MODE";
        chip.classList.toggle("play-hidden", hideInPlay);
      }
    }

    function applyPreset(name) {
      if (name === "a1Full") {
        setScript("train");
        app.currentTrainPreset = "a1Full";
        app.core.task = "a1_wmp_amp_terrain";
        app.core.load_run = "";
        app.core.checkpoint = "";
        setEnvText("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True");
        resetValuesForScript("train");
        setValue("--runner", "wmp_amp");
        setValue("--headless", true);
        setValue("--enable_cameras", true);
        setValue("--num_envs", "4096");
        setValue("--wmp_camera_num_envs", "1024");
        setValue("--max_iterations", "20000");
        setValue("--num_steps_per_env", "24");
        setValue("--logger", "wandb");
        setValue("--log_project_name", "a1_wmp_amp");
        setValue("--run_name", "a1_wmp_amp_4096env_1024cam_original_v1");
      }
      if (name === "a1Smoke") {
        setScript("train");
        app.currentTrainPreset = "a1Smoke";
        app.core.task = "a1_wmp_amp_terrain";
        clearCheckpointSelection();
        setEnvText("");
        resetValuesForScript("train");
        setValue("--runner", "wmp_amp");
        setValue("--headless", true);
        setValue("--enable_cameras", true);
        setValue("--num_envs", "2");
        setValue("--wmp_camera_num_envs", "2");
        setValue("--max_iterations", "1");
        setValue("--num_steps_per_env", "2");
        setValue("--num_mini_batches", "1");
        setValue("--wmp_depth_training_iters", "1");
        setValue("--wmp_depth_batch_size", "2");
        setValue("--wmp_train_steps_per_iter", "1");
        setValue("--logger", "tensorboard");
        setValue("--run_name", "a1_wmp_web_smoke");
      }
      if (name === "playFlat") {
        setValue("--runner", "wmp_amp");
        setValue("--num_envs", "4");
        setValue("--play_flat", true);
      }
      if (name === "depthImage") {
        setValue("--runner", "wmp_amp");
        setValue("--num_envs", "4");
        setValue("--show_depth_image", true);
        setValue("--depth_image_mode", "auto");
        setValue("--depth_image_save_interval", "5");
        setValue("--camera_offset_pos", "0.27 0.0 0.10");
        setValue("--camera_disable_random_rotation", true);
      }
      if (name === "depthPoints") {
        setValue("--runner", "wmp_amp");
        setValue("--num_envs", "4");
        setValue("--show_depth_points", true);
        setValue("--show_camera_axes", true);
        setValue("--depth_point_camera_index", "-1");
        setValue("--depth_point_stride", "1");
        setValue("--depth_point_max", "20000");
        setValue("--depth_point_forward_min", "0.0");
        setValue("--depth_point_forward_max", "3.0");
        setValue("--depth_point_debug", true);
        setValue("--depth_point_draw_rays", false);
        setValue("--camera_offset_pos", "0.27 0.0 0.10");
        setValue("--camera_disable_random_rotation", true);
      }
      syncCoreInputs();
      renderOptions();
      updatePreviewSoon();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.error || response.statusText);
      return data;
    }

    async function startProcess() {
      try {
        await postJson("/api/start", collectPayload());
        await pollLogs();
      } catch (err) {
        appendLocalLog("[ERROR] " + err.message + "\n");
      }
    }

    async function stopProcess() {
      try {
        await postJson("/api/stop", {});
        await pollLogs();
      } catch (err) {
        appendLocalLog("[ERROR] " + err.message + "\n");
      }
    }

    function appendLocalLog(text) {
      const pane = $("logPane");
      pane.textContent += text;
      pane.scrollTop = pane.scrollHeight;
    }

    async function pollLogs() {
      try {
        const result = await fetch(`/api/logs?since=${app.latestLog}`).then((r) => r.json());
        for (const line of result.lines) {
          app.latestLog = Math.max(app.latestLog, line.seq);
          appendLocalLog(line.text);
        }
        updateStatus(result);
      } catch (err) {
        updateStatus({ running: false, pid: null });
      }
    }

    function updateStatus(result) {
      const running = Boolean(result.running);
      $("runDot").classList.toggle("running", running);
      $("runState").textContent = running ? "运行中" : "未运行";
      $("pidState").textContent = running && result.pid ? `PID ${result.pid}` : "PID -";
      $("startBtn").disabled = running;
      $("stopBtn").disabled = !running;
    }

    async function boot() {
      app.data = await fetch("/api/state").then((r) => r.json());
      $("pythonPath").textContent = app.data.python;
      ensureDefaults("train");
      ensureDefaults("play");
      renderTasks();
      renderCheckpoints();
      setScript("train");
      applyPreset("a1Full");
      updateStatus(app.data);
      $("scriptTrain").addEventListener("click", () => setScript("train"));
      $("scriptPlay").addEventListener("click", () => setScript("play"));
      $("taskSelect").addEventListener("change", (e) => { app.core.task = e.target.value; updatePreviewSoon(); });
      $("envInput").addEventListener("input", (e) => { app.env = e.target.value; syncEnvChips(); updatePreviewSoon(); });
      $("extraInput").addEventListener("input", (e) => { app.extra = e.target.value; updatePreviewSoon(); });
      $("filterInput").addEventListener("input", applyFilter);
      $("resetDefaults").addEventListener("click", () => { app.values[app.script] = {}; ensureDefaults(app.script); renderOptions(); updatePreviewSoon(); });
      $("startBtn").addEventListener("click", startProcess);
      $("stopBtn").addEventListener("click", stopProcess);
      $("clearLogBtn").addEventListener("click", () => { $("logPane").textContent = ""; });
      $("checkpointSelect").addEventListener("change", (event) => {
        const path = event.target.value;
        const ckpt = app.data.checkpoints.find((item) => item.path === path);
        setCheckpointFromItem(ckpt);
        syncCoreInputs();
        updatePreviewSoon();
      });
      for (const chip of document.querySelectorAll("[data-env]")) {
        chip.addEventListener("click", () => toggleEnvToken(chip.dataset.env));
      }
      setInterval(pollLogs, 1000);
      updatePreviewSoon();
    }

    boot().catch((err) => {
      document.body.innerHTML = `<pre>${err.stack || err}</pre>`;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
