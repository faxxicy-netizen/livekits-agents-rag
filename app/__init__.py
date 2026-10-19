"""Knowledge-base voice assistant application."""

# Windows ships an older onnxruntime.dll in System32. Load the version installed
# in this virtual environment before sherpa's extension resolves that DLL name.
# This changes only this process; it does not modify Windows or the system PATH.
import importlib.util
import sys
from pathlib import Path

if sys.platform == "win32":
    import ctypes

    _spec = importlib.util.find_spec("onnxruntime")
    if _spec and _spec.origin:
        _library = Path(_spec.origin).parent / "capi/onnxruntime.dll"
        if _library.is_file():
            _ort_dll = ctypes.WinDLL(str(_library))
