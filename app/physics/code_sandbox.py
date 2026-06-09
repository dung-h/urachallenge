from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ALLOWED_IMPORTS = {"math", "statistics", "decimal", "fractions"}
BLOCKED_CALLS = {
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
    "__import__",
}
BLOCKED_NAMES = {
    "breakpoint",
    "exit",
    "help",
    "memoryview",
    "quit",
}


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class CodeSafetyError(ValueError):
    pass


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        current: ast.AST | None = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
    return None


def validate_python_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise CodeSafetyError(f"syntax_error:{exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name.split(".", 1)[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                if node.module is None:
                    raise CodeSafetyError("relative_import_blocked")
                names = [node.module.split(".", 1)[0]]
            blocked = sorted(set(names) - ALLOWED_IMPORTS)
            if blocked:
                raise CodeSafetyError(f"import_blocked:{','.join(blocked)}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise CodeSafetyError("dunder_attribute_blocked")
        elif isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in BLOCKED_NAMES):
            raise CodeSafetyError(f"name_blocked:{node.id}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                root_name = name.split(".", 1)[0]
                if name in BLOCKED_CALLS or root_name in BLOCKED_CALLS:
                    raise CodeSafetyError(f"call_blocked:{name}")


def run_python_code(code: str, timeout: float = 3.0) -> SandboxResult:
    try:
        validate_python_code(code)
    except CodeSafetyError as exc:
        return SandboxResult(ok=False, error=str(exc))

    with tempfile.TemporaryDirectory(prefix="ura_physics_sandbox_") as temp_dir:
        script_path = Path(temp_dir) / "solution.py"
        script_path.write_text(code, encoding="utf-8")
        env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(ok=False, error="timeout")
        except OSError as exc:
            return SandboxResult(ok=False, error=f"os_error:{type(exc).__name__}")

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode != 0:
        return SandboxResult(ok=False, stdout=stdout, stderr=stderr, error=f"nonzero_exit:{result.returncode}")
    return SandboxResult(ok=True, stdout=stdout, stderr=stderr)
