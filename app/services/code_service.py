import ast
import subprocess
import sys
import tempfile
from pathlib import Path


MAX_CODE_CHARS = 10_000
MAX_OUTPUT_CHARS = 5_000


def validate_python(code: str) -> dict:
    """Check Python syntax before attempting execution."""
    if not isinstance(code, str) or not code.strip():
        return {
            "valid": False,
            "error": "Enter some Python code first.",
        }

    if len(code) > MAX_CODE_CHARS:
        return {
            "valid": False,
            "error": f"Code exceeds the {MAX_CODE_CHARS}-character limit.",
        }

    try:
        ast.parse(code)
    except SyntaxError as error:
        return {
            "valid": False,
            "error": (
                f"SyntaxError on line {error.lineno}: {error.msg}"
            ),
        }

    return {"valid": True, "error": None}


def run_python(code: str, timeout: int = 3) -> dict:
    """
    Execute a Python snippet in a temporary subprocess.

    WARNING: This is NOT a security sandbox. Use locally with trusted
    code only. Do not expose this function to untrusted public users.
    """
    if not 1 <= timeout <= 10:
        raise ValueError("timeout must be between 1 and 10 seconds")

    validation = validate_python(code)

    if not validation["valid"]:
        return {
            "success": False,
            "stdout": "",
            "stderr": validation["error"],
            "returncode": None,
            "error_type": "SyntaxError",
        }

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(code)
            temp_path = Path(handle.name)

        result = subprocess.run(
            [sys.executable, "-I", str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=temp_path.parent,
            check=False,
        )

        stdout = result.stdout[-MAX_OUTPUT_CHARS:]
        stderr = result.stderr[-MAX_OUTPUT_CHARS:]

        return {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
            "error_type": (
                None if result.returncode == 0
                else "RuntimeError"
            ),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution exceeded the {timeout}-second timeout.",
            "returncode": None,
            "error_type": "Timeout",
        }

    except OSError as error:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(error),
            "returncode": None,
            "error_type": type(error).__name__,
        }

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
