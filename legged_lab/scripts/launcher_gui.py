"""可视化启动器：用于填写 train.py / play.py 参数并一键启动。"""

from __future__ import annotations

import ast
import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText


@dataclass
class OptionSpec:
    """命令行参数定义。"""

    name: str
    takes_value: bool
    value_hint: str = ""
    default_value: str = ""


@dataclass
class OptionWidget:
    """参数控件绑定。"""

    spec: OptionSpec
    row: ttk.Frame
    value_var: tk.StringVar | None = None
    flag_var: tk.BooleanVar | None = None


class LauncherGUI:
    """LeggedLab 训练与回放启动器。"""

    CORE_OPTIONS = {"--task", "--load_run", "--checkpoint"}

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LeggedLab Launcher")
        self.root.geometry("1220x780")

        self.repo_root = Path(__file__).resolve().parents[2]
        self.scripts = {
            "train": self.repo_root / "legged_lab" / "scripts" / "train.py",
            "play": self.repo_root / "legged_lab" / "scripts" / "play.py",
        }

        self.task_choices = self._detect_tasks()
        self.latest_checkpoint_info = self._find_latest_checkpoint_info()

        self.option_specs: dict[str, list[OptionSpec]] = {"train": [], "play": []}
        self.option_defaults: dict[str, dict[str, str]] = {"train": {}, "play": {}}
        self.option_widgets: dict[str, list[OptionWidget]] = {"train": [], "play": []}
        self.option_widget_map: dict[str, dict[str, OptionWidget]] = {"train": {}, "play": {}}

        self.process: subprocess.Popen | None = None
        self.reader_thread: threading.Thread | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._configure_style()
        self._build_layout()
        self._load_options_and_render()
        self._poll_logs()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        """设置更清爽的字体与控件样式。"""
        style = ttk.Style(self.root)
        theme_names = set(style.theme_names())
        for preferred_theme in ["vista", "xpnative", "aqua", "alt", "clam", "default", "classic"]:
            if preferred_theme in theme_names:
                style.theme_use(preferred_theme)
                break

        try:
            # 适度放大 DPI，降低“像素风”观感
            self.root.tk.call("tk", "scaling", 1.2)
        except Exception:
            pass

        # 用户指定：统一使用 Ubuntu 字体族
        ui_family = "Ubuntu" if "Ubuntu" in set(tkfont.families(self.root)) else self._pick_font_family(["Ubuntu"])
        mono_family = (
            "Ubuntu Mono"
            if "Ubuntu Mono" in set(tkfont.families(self.root))
            else self._pick_font_family(["Ubuntu Mono", "DejaVu Sans Mono"], fallback="TkFixedFont")
        )

        # 强制覆盖 Tk 命名字体，避免回退到位图字体
        for named in ["TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"]:
            try:
                tkfont.nametofont(named).configure(family=ui_family, size=12)
            except Exception:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=10)
        except Exception:
            pass

        self.ui_font = (ui_family, 12)
        self.ui_small_font = (ui_family, 11)
        self.ui_bold_font = (ui_family, 12, "bold")
        self.mono_font = (mono_family, 10)

        self.root.option_add("*Font", self.ui_font)

        style.configure("TLabel", font=self.ui_font)
        style.configure("TButton", font=self.ui_font)
        style.configure("TEntry", font=self.ui_font)
        style.configure("TCombobox", font=self.ui_font)
        style.configure("TLabelframe.Label", font=self.ui_bold_font)
        style.configure("Hint.TLabel", font=self.ui_small_font, foreground="#4f5b66")
        style.configure("Preview.TLabel", font=self.ui_small_font, foreground="#0F4C81")

    def _pick_font_family(self, preferred: list[str], fallback: str = "TkDefaultFont") -> str:
        available = set(tkfont.families(self.root))
        for name in preferred:
            if name in available:
                return name
        try:
            return tkfont.nametofont(fallback).actual("family")
        except Exception:
            return fallback

    def _detect_tasks(self) -> list[str]:
        """从 envs 注册文件提取任务名，用于 task 下拉框。"""
        env_init = self.repo_root / "legged_lab" / "envs" / "__init__.py"
        if not env_init.exists():
            return []

        try:
            text = env_init.read_text(encoding="utf-8")
        except Exception:
            return []

        tasks = re.findall(r'task_registry\.register\("([a-zA-Z0-9_]+)"', text)
        seen: set[str] = set()
        ordered_tasks: list[str] = []
        for task in tasks:
            if task not in seen:
                ordered_tasks.append(task)
                seen.add(task)
        return ordered_tasks

    def _find_latest_checkpoint_info(self) -> dict[str, str] | None:
        """查找 logs 下最新 checkpoint，用于核心参数默认值。"""
        logs_dir = self.repo_root / "logs"
        if not logs_dir.exists():
            return None

        checkpoints = list(logs_dir.glob("*/*/*.pt"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        try:
            rel = latest.resolve().relative_to(logs_dir.resolve())
        except Exception:
            return None

        parts = rel.parts
        if len(parts) < 3:
            return None

        return {
            "task": parts[0],
            "run": parts[1],
            "name": latest.name,
            "path": str(latest),
        }

    def _extract_default_values(self, script_path: Path) -> dict[str, str]:
        """从 add_argument(default=...) 提取默认值。"""
        defaults: dict[str, str] = {}
        paths = [script_path, self.repo_root / "legged_lab" / "utils" / "cli_args.py"]
        for path in paths:
            if not path.exists():
                continue

            try:
                text = path.read_text(encoding="utf-8")
                tree = ast.parse(text)
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

                default_value = self._eval_ast_default(default_node, constants)
                defaults[option_name] = self._default_to_text(default_value)

        return defaults

    def _collect_module_constants(self, tree: ast.AST) -> dict[str, object]:
        constants: dict[str, object] = {}
        for node in tree.body if isinstance(tree, ast.Module) else []:
            if isinstance(node, ast.Assign):
                value = self._eval_ast_default(node.value, constants)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
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
            # 兼容 os.environ.get("X", "default")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if len(node.args) >= 2:
                    return self._eval_ast_default(node.args[1], constants)
        return None

    def _default_to_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(container)
        top.pack(fill=tk.X)

        ttk.Label(top, text="脚本:").pack(side=tk.LEFT)
        self.script_var = tk.StringVar(value="train")
        self.script_combo = ttk.Combobox(
            top,
            textvariable=self.script_var,
            values=["train", "play"],
            state="readonly",
            width=10,
        )
        self.script_combo.pack(side=tk.LEFT, padx=(6, 14))
        self.script_combo.bind("<<ComboboxSelected>>", self._on_script_change)

        ttk.Label(top, text="参数过滤:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_entry = ttk.Entry(top, textvariable=self.filter_var, width=24)
        self.filter_entry.pack(side=tk.LEFT, padx=(6, 12))
        self.filter_var.trace_add("write", lambda *_: self._apply_filter())

        self.refresh_btn = ttk.Button(top, text="刷新参数", command=self._refresh_options)
        self.refresh_btn.pack(side=tk.LEFT)

        self.command_preview_var = tk.StringVar(value="命令预览: ")
        ttk.Label(container, textvariable=self.command_preview_var, style="Preview.TLabel").pack(fill=tk.X, pady=(10, 6))

        main = ttk.Panedwindow(container, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self._build_scrollable_form(left)
        self._build_controls(right)

    def _build_scrollable_form(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="参数列表", padding=8)
        frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.form_inner = ttk.Frame(self.canvas)
        self.form_window = self.canvas.create_window((0, 0), window=self.form_inner, anchor="nw")

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.form_inner.bind("<Configure>", self._on_form_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _build_controls(self, parent: ttk.Frame) -> None:
        cmd_frame = ttk.LabelFrame(parent, text="启动控制", padding=8)
        cmd_frame.pack(fill=tk.X)

        ttk.Label(cmd_frame, text="额外参数（原样拼接）:").pack(anchor=tk.W)
        self.extra_args_var = tk.StringVar()
        extra_entry = ttk.Entry(cmd_frame, textvariable=self.extra_args_var)
        extra_entry.pack(fill=tk.X, pady=(4, 8))
        self.extra_args_var.trace_add("write", lambda *_: self._update_preview())

        button_row = ttk.Frame(cmd_frame)
        button_row.pack(fill=tk.X)

        self.run_btn = ttk.Button(button_row, text="启动当前脚本", command=self._start_current)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.play_btn = ttk.Button(button_row, text="快速 Play", command=lambda: self._switch_and_start("play"))
        self.play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        self.stop_btn = ttk.Button(button_row, text="停止", command=self._stop)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(
            cmd_frame,
            text="提示: 未填写的参数不会传入命令，脚本会使用默认值。",
            style="Hint.TLabel",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))

        log_frame = ttk.LabelFrame(parent, text="运行日志", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, font=self.mono_font)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "[INFO] 启动器已就绪。\n")
        self.log_text.configure(state=tk.DISABLED)

    def _load_options_and_render(self) -> None:
        for script_name, script_path in self.scripts.items():
            defaults = self._extract_default_values(script_path)
            specs = self._extract_options(script_path)
            for spec in specs:
                spec.default_value = defaults.get(spec.name, "")
            self.option_defaults[script_name] = defaults
            self.option_specs[script_name] = specs
        self._render_options(self.script_var.get())

    def _refresh_options(self) -> None:
        self._append_log("[INFO] 正在刷新参数...\n")
        current = self.script_var.get()
        self.option_specs[current] = self._extract_options(self.scripts[current])
        self._render_options(current)

    def _extract_options(self, script_path: Path) -> list[OptionSpec]:
        cmd = [sys.executable, str(script_path), "--help"]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except Exception as exc:
            self._append_log(f"[ERROR] 拉取参数失败: {exc}\n")
            return []

        if result.returncode not in (0, 1):
            self._append_log(
                f"[WARN] 解析 {script_path.name} 参数失败，返回码={result.returncode}。\n{result.stdout}\n"
            )
            return self._extract_options_from_source(script_path)

        specs: list[OptionSpec] = []
        seen: set[str] = set()

        for line in result.stdout.splitlines():
            match = re.match(r"^\s{2,}(.+?)\s{2,}.*$", line)
            if not match:
                continue
            invocation = match.group(1).strip()
            if "--" not in invocation:
                continue

            pieces = [piece.strip() for piece in invocation.split(",")]
            long_piece = next((piece for piece in pieces if piece.startswith("--")), "")
            if not long_piece:
                continue

            tokens = long_piece.split()
            if not tokens:
                continue

            option_name = tokens[0]
            if option_name in {"--help", "-h"}:
                continue
            if option_name in seen:
                continue

            value_hint = " ".join(tokens[1:]).strip()
            takes_value = bool(value_hint)
            specs.append(OptionSpec(name=option_name, takes_value=takes_value, value_hint=value_hint))
            seen.add(option_name)

        if specs:
            return specs

        self._append_log(
            f"[WARN] 未从 {script_path.name} --help 提取到参数，已回退到源码解析（不含 AppLauncher 动态参数）。\n"
        )
        return self._extract_options_from_source(script_path)

    def _extract_options_from_source(self, script_path: Path) -> list[OptionSpec]:
        """回退解析：从源码中提取 add_argument 参数。"""
        try:
            script_text = script_path.read_text(encoding="utf-8")
            cli_text = (self.repo_root / "legged_lab" / "utils" / "cli_args.py").read_text(encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[ERROR] 回退解析失败: {exc}\n")
            return []

        option_pattern = re.compile(r"add_argument\(\s*[\"'](--[a-zA-Z0-9_-]+)[\"']")
        type_pattern = re.compile(
            r"add_argument\(\s*[\"'](--[a-zA-Z0-9_-]+)[\"'].*?type\s*=\s*[a-zA-Z_][a-zA-Z0-9_]*",
            re.S,
        )

        candidates = option_pattern.findall(script_text) + option_pattern.findall(cli_text)
        with_type = set(type_pattern.findall(script_text) + type_pattern.findall(cli_text))

        specs: list[OptionSpec] = []
        seen: set[str] = set()
        for option_name in candidates:
            if option_name in seen or option_name == "--help":
                continue
            takes_value = option_name in with_type
            specs.append(OptionSpec(name=option_name, takes_value=takes_value))
            seen.add(option_name)
        return specs

    def _render_options(self, script_name: str) -> None:
        for child in self.form_inner.winfo_children():
            child.destroy()

        self.option_widgets[script_name] = []
        self.option_widget_map[script_name] = {}

        specs = self.option_specs.get(script_name, [])
        if not specs:
            ttk.Label(self.form_inner, text="未获取到参数，请先检查 IsaacLab 环境或点击刷新参数。", foreground="#A04545").pack(
                anchor=tk.W
            )
            self._update_preview()
            return

        self._build_core_args_row(script_name)

        for spec in specs:
            if spec.name in self.CORE_OPTIONS:
                continue

            row = ttk.Frame(self.form_inner)
            row.pack(fill=tk.X, pady=2)

            if spec.takes_value:
                ttk.Label(row, text=spec.name, width=22).pack(side=tk.LEFT)
                value_var = tk.StringVar()
                if spec.default_value:
                    value_var.set(spec.default_value)
                entry = ttk.Entry(row, textvariable=value_var, width=22)
                entry.pack(side=tk.LEFT, padx=(0, 8))
                value_var.trace_add("write", lambda *_: self._update_preview())
                widget = OptionWidget(spec=spec, row=row, value_var=value_var)

                if spec.value_hint:
                    ttk.Label(row, text=spec.value_hint, foreground="#666666", width=18).pack(side=tk.LEFT)
            else:
                flag_var = tk.BooleanVar(value=(spec.default_value.lower() == "true"))
                chk = ttk.Checkbutton(row, text=spec.name, variable=flag_var, command=self._update_preview)
                chk.pack(side=tk.LEFT, anchor=tk.W)
                widget = OptionWidget(spec=spec, row=row, flag_var=flag_var)

            self.option_widgets[script_name].append(widget)
            self.option_widget_map[script_name][spec.name] = widget

        self._apply_filter()
        self._update_preview()

    def _build_core_args_row(self, script_name: str) -> None:
        """第一行 task，第二行 checkpoint（含隐藏 load_run）。"""
        default_task = ""
        default_checkpoint_path = ""
        default_load_run = ""
        if self.latest_checkpoint_info is not None:
            default_task = self.latest_checkpoint_info.get("task", "")
            default_checkpoint_path = self.latest_checkpoint_info.get("path", "")
            default_load_run = self.latest_checkpoint_info.get("run", "")
        elif self.task_choices:
            default_task = self.task_choices[0]

        row1 = ttk.Frame(self.form_inner)
        row1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row1, text="task").pack(side=tk.LEFT)
        task_var = tk.StringVar(value=default_task)
        task_combo = ttk.Combobox(row1, textvariable=task_var, values=self.task_choices, state="normal", width=18)
        task_combo.pack(side=tk.LEFT, padx=(4, 10))
        task_var.trace_add("write", lambda *_: self._update_preview())

        row2 = ttk.Frame(self.form_inner)
        row2.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(row2, text="checkpoint").pack(side=tk.LEFT)
        checkpoint_var = tk.StringVar(value=default_checkpoint_path)
        checkpoint_entry = ttk.Entry(row2, textvariable=checkpoint_var, width=42)
        checkpoint_entry.pack(side=tk.LEFT, padx=(4, 6))
        checkpoint_var.trace_add("write", lambda *_: self._update_preview())

        browse_btn = ttk.Button(row2, text="浏览...", width=8, command=lambda: self._choose_checkpoint(checkpoint_var))
        browse_btn.pack(side=tk.LEFT)

        # load_run 改为隐藏字段，仅用于命令拼接
        run_var = tk.StringVar(value=default_load_run)

        task_widget = OptionWidget(spec=OptionSpec("--task", True, default_value=default_task), row=row1, value_var=task_var)
        run_widget = OptionWidget(spec=OptionSpec("--load_run", True, default_value=default_load_run), row=row2, value_var=run_var)
        ckpt_widget = OptionWidget(
            spec=OptionSpec("--checkpoint", True, default_value=default_checkpoint_path),
            row=row2,
            value_var=checkpoint_var,
        )

        self.option_widgets[script_name].extend([task_widget, run_widget, ckpt_widget])
        self.option_widget_map[script_name]["--task"] = task_widget
        self.option_widget_map[script_name]["--load_run"] = run_widget
        self.option_widget_map[script_name]["--checkpoint"] = ckpt_widget

    def _choose_checkpoint(self, checkpoint_var: tk.StringVar) -> None:
        path = self._open_checkpoint_dialog()
        if not path:
            return

        checkpoint_path = Path(path)
        checkpoint_var.set(str(checkpoint_path))
        self._autofill_from_checkpoint_path(checkpoint_path)
        self._update_preview()

    def _open_checkpoint_dialog(self) -> str:
        initial_dir = self._guess_checkpoint_initial_dir()
        return filedialog.askopenfilename(
            title="选择 checkpoint",
            initialdir=str(initial_dir),
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
        )

    def _guess_checkpoint_initial_dir(self) -> Path:
        logs_dir = self.repo_root / "logs"
        if not logs_dir.exists():
            return self.repo_root

        task_value = self._get_option_value("--task")
        run_value = self._get_option_value("--load_run")

        if task_value and run_value:
            candidate = logs_dir / task_value / run_value
            if candidate.exists():
                return candidate
        if task_value:
            candidate = logs_dir / task_value
            if candidate.exists():
                return candidate
        return logs_dir

    def _autofill_from_checkpoint_path(self, checkpoint_path: Path) -> None:
        """根据 logs 路径自动回填 load_run/task。"""
        try:
            rel = checkpoint_path.resolve().relative_to((self.repo_root / "logs").resolve())
        except Exception:
            return

        parts = rel.parts
        if len(parts) >= 3:
            experiment_name = parts[0]
            run_name = parts[1]

            self._set_option_value("--load_run", run_name)
            self._set_option_value("--task", experiment_name)

    def _resolve_checkpoint_input(self, raw_value: str) -> tuple[str, str | None, str | None]:
        """将 checkpoint 输入值解析为 CLI 需要的 checkpoint/load_run/task。"""
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

        parts = rel.parts
        if len(parts) >= 3:
            return ckpt_name, parts[1], parts[0]
        return ckpt_name, None, None

    def _get_option_value(self, option_name: str, script_name: str | None = None) -> str:
        target_script = script_name or self.script_var.get()
        widget = self.option_widget_map.get(target_script, {}).get(option_name)
        if widget is None or widget.value_var is None:
            return ""
        return widget.value_var.get().strip()

    def _set_option_value(self, option_name: str, value: str, script_name: str | None = None) -> None:
        target_script = script_name or self.script_var.get()
        widget = self.option_widget_map.get(target_script, {}).get(option_name)
        if widget is None or widget.value_var is None:
            return
        widget.value_var.set(value)

    def _collect_args(self, script_name: str) -> list[str]:
        args: list[str] = []
        task_value = self._get_option_value("--task", script_name)
        load_run_value = self._get_option_value("--load_run", script_name)
        checkpoint_raw = self._get_option_value("--checkpoint", script_name)
        checkpoint_value, run_from_path, task_from_path = self._resolve_checkpoint_input(checkpoint_raw)

        if run_from_path:
            load_run_value = run_from_path
            self._set_option_value("--load_run", run_from_path, script_name)
        if task_from_path and (not task_value):
            task_value = task_from_path
            self._set_option_value("--task", task_from_path, script_name)

        if task_value:
            args.append(f"--task={task_value}")
        if checkpoint_value:
            if load_run_value:
                args.append(f"--load_run={load_run_value}")
            args.append(f"--checkpoint={checkpoint_value}")

        for widget in self.option_widgets.get(script_name, []):
            spec = widget.spec
            if spec.name in self.CORE_OPTIONS:
                continue

            if spec.takes_value:
                if widget.value_var is None:
                    continue
                value = widget.value_var.get().strip()
                if value and value != spec.default_value:
                    args.append(f"{spec.name}={value}")
            else:
                if widget.flag_var is not None and widget.flag_var.get():
                    args.append(spec.name)

        extra = self.extra_args_var.get().strip()
        if extra:
            try:
                args.extend(shlex.split(extra))
            except ValueError as exc:
                self._append_log(f"[WARN] 额外参数解析失败: {exc}\n")
        return args

    def _build_command(self, script_name: str) -> list[str]:
        script_path = self.scripts[script_name]
        cmd = [sys.executable, str(script_path)]
        cmd.extend(self._collect_args(script_name))
        return cmd

    def _update_preview(self) -> None:
        script_name = self.script_var.get()
        cmd = self._build_command(script_name)
        self.command_preview_var.set("命令预览: " + " ".join(shlex.quote(item) for item in cmd))

    def _start(self, script_name: str) -> None:
        if self.process is not None and self.process.poll() is None:
            self._append_log("[WARN] 已有任务在运行，请先停止。\n")
            return

        cmd = self._build_command(script_name)
        self._append_log(f"[INFO] 启动 {script_name}: {' '.join(shlex.quote(item) for item in cmd)}\n")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
        except Exception as exc:
            self._append_log(f"[ERROR] 启动失败: {exc}\n")
            self.process = None
            return

        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _start_current(self) -> None:
        self._start(self.script_var.get())

    def _switch_and_start(self, script_name: str) -> None:
        if self.script_var.get() != script_name:
            self.script_var.set(script_name)
            self._render_options(script_name)
        self._start(script_name)

    def _read_process_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            self.log_queue.put(line)
        rc = self.process.wait()
        self.log_queue.put(f"\n[INFO] 进程结束，返回码: {rc}\n")

    def _stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self._append_log("[INFO] 当前无运行中的进程。\n")
            return

        self._append_log("[INFO] 正在停止进程...\n")
        try:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except Exception as exc:
            self._append_log(f"[WARN] 停止失败: {exc}\n")

    def _poll_logs(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(line)
        self.root.after(120, self._poll_logs)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _apply_filter(self) -> None:
        script_name = self.script_var.get()
        keyword = self.filter_var.get().strip().lower()
        row_visibility: dict[ttk.Frame, bool] = {}
        for widget in self.option_widgets.get(script_name, []):
            show = True
            if keyword:
                show = keyword in widget.spec.name.lower() or keyword in widget.spec.value_hint.lower()
            row_visibility[widget.row] = row_visibility.get(widget.row, False) or show

        for row, show in row_visibility.items():
            if show:
                row.pack(fill=tk.X, pady=2)
            else:
                row.pack_forget()

    def _on_script_change(self, _event=None) -> None:
        script_name = self.script_var.get()
        self._render_options(script_name)

    def _on_form_configure(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event=None) -> None:
        if event is not None:
            self.canvas.itemconfigure(self.form_window, width=event.width)

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self._stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    LauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
