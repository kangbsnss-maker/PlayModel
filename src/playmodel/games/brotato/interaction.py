"""Read-only Win32 target identity. Foreground calibration is retired.

BackgroundController reuses these bindings. No binding activates a window,
moves the physical pointer, or emits global input.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import os
from pathlib import Path


class WindowController:
    def __init__(self, hwnd: int, expected_exe: Path):
        if os.name != "nt":
            raise OSError("Windows input is required")
        self.hwnd = hwnd
        self.expected = os.path.normcase(str(expected_exe.resolve()))
        self.user = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)

        def api(dll, name, args, result):
            fn = getattr(dll, name)
            fn.argtypes, fn.restype = args, result
            return fn

        self.pid = api(self.user, "GetWindowThreadProcessId", [w.HWND, ctypes.POINTER(w.DWORD)], w.DWORD)
        self.open_process = api(self.kernel, "OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE)
        self.close_handle = api(self.kernel, "CloseHandle", [w.HANDLE], w.BOOL)
        self.image_name = api(self.kernel, "QueryFullProcessImageNameW", [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)], w.BOOL)
        self.foreground = api(self.user, "GetForegroundWindow", [], w.HWND)
        self.is_iconic = api(self.user, "IsIconic", [w.HWND], w.BOOL)
        self.rect = api(self.user, "GetClientRect", [w.HWND, ctypes.POINTER(w.RECT)], w.BOOL)
        self.get_cursor = api(self.user, "GetCursorPos", [ctypes.POINTER(w.POINT)], w.BOOL)
        self.key_state = api(self.user, "GetAsyncKeyState", [ctypes.c_int], ctypes.c_short)
        self.dpi = api(self.user, "SetProcessDpiAwarenessContext", [w.HANDLE], w.BOOL)
        self.dpi(ctypes.c_void_p(-4))
        self.identity = self._identity()
        if self.identity[1] != self.expected:
            raise OSError("Input target executable does not match Brotato")

    def _identity(self):
        pid = w.DWORD()
        if not self.pid(self.hwnd, ctypes.byref(pid)):
            raise OSError("Game window no longer exists")
        process = self.open_process(0x1000, False, pid.value)
        if not process:
            raise OSError("Could not verify game process")
        try:
            buffer, size = ctypes.create_unicode_buffer(32768), w.DWORD(32768)
            if not self.image_name(process, 0, buffer, ctypes.byref(size)):
                raise OSError("Could not verify game executable")
            return pid.value, self._canonical_image_path(buffer.value)
        finally:
            self.close_handle(process)

    def _canonical_image_path(self, raw_path):
        # Query the process on every check, but do not repeatedly walk the
        # filesystem for the exact same OS-reported executable name. PID and
        # changed image names still undergo their original identity checks.
        cached = getattr(self, '_image_path_cache', None)
        if cached is not None and cached[0] == raw_path:
            return cached[1]
        canonical = os.path.normcase(str(Path(raw_path).resolve()))
        self._image_path_cache = raw_path, canonical
        return canonical

    def _retired(self, *args, **kwargs):
        raise OSError("Foreground input retired: use HWND-only BackgroundController")

    activate = hover = click = set_movement = release = _retired


if __name__ == "__main__":
    raise SystemExit("Foreground calibration retired; use background keyboard calibration")
