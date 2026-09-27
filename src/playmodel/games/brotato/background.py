"""Experimental HWND-only input. Posting is not proof that Brotato applied it.

This module never activates a window, moves the system cursor, or sends global
keyboard/mouse input. Each input capability must be verified in the actual game.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes as w
from pathlib import Path

from .interaction import WindowController


class BackgroundController:
    def __init__(self, hwnd: int, expected_exe: Path):
        # Reuse process identity and Win32 bindings only; never call foreground
        # activation, global hover/click, or SendInput from the context.
        self._context = WindowController(hwnd, expected_exe)
        self.hwnd = hwnd
        self.held: set[int] = set()
        self._post = self._context.user.PostMessageW
        self._post.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM]
        self._post.restype = w.BOOL
        self._scan = self._context.user.MapVirtualKeyW
        self._scan.argtypes, self._scan.restype = [w.UINT, w.UINT], w.UINT

    def check(self):
        if self._context.key_state(0x77) & 0x8000:
            raise OSError("Stopped by F8")
        if self._context._identity() != self._context.identity or self._context.is_iconic(self.hwnd):
            raise OSError("Game identity changed or game was minimized")

    def _position(self, x: int, y: int, expected_size: tuple[int, int]) -> int:
        self.check()
        rect = w.RECT()
        if not self._context.rect(self.hwnd, ctypes.byref(rect)):
            raise OSError("Could not verify game bounds")
        size = rect.right - rect.left, rect.bottom - rect.top
        if size != expected_size or not (0 <= x < min(size[0], 32768) and 0 <= y < min(size[1], 32768)):
            raise OSError("Background input coordinates no longer match the frame")
        return (y << 16) | x

    def _message(self, message: int, key: int, data: int):
        if not self._post(self.hwnd, message, key, data):
            raise ctypes.WinError(ctypes.get_last_error())

    def hover(self, x: int, y: int, expected_size: tuple[int, int]):
        raise OSError("Mouse messages are unverified; use background keyboard navigation")

    def click(self, x: int, y: int, expected_size: tuple[int, int]):
        raise OSError("Mouse messages are unverified; use background keyboard navigation")

    def _key(self, key: int, up: bool):
        data = 1 | (self._scan(key, 0) << 16)
        if key in (0x25, 0x26, 0x27, 0x28):
            data |= 1 << 24  # Dedicated arrow keys, not keypad arrows.
        if up:
            data |= (1 << 30) | (1 << 31)
        self._message(0x0101 if up else 0x0100, key, data)

    def set_movement(self, keys: set[int]):
        if not keys <= {0x57, 0x41, 0x53, 0x44}:
            raise ValueError("Background movement allows WASD only")
        self.check()
        for key in self.held - keys:
            self._key(key, True)
            self.held.remove(key)
        for key in keys - self.held:
            self._key(key, False)
            self.held.add(key)

    def tap_menu(self, key: str):
        keys = {"left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28, "enter": 0x0D, "escape": 0x1B, "tab": 0x09, "reroll": 0x46}
        if key not in keys:
            raise ValueError("Unsupported menu key")
        self.check()
        self._key(keys[key], False)
        try:
            self._key(keys[key], True)
        except OSError:
            self._key(keys[key], True)
            raise

    def release(self):
        errors = []
        for key in tuple(self.held):
            try:
                self._key(key, True)
                self.held.remove(key)
            except OSError as error:
                errors.append(error)
        if errors:
            raise OSError("Some owned background keys could not be released") from errors[0]

    def desktop_state(self):
        point = w.POINT()
        if not self._context.get_cursor(ctypes.byref(point)):
            raise OSError("Could not observe cursor position")
        return {"cursor": [point.x, point.y], "foreground_hwnd": self._context.foreground()}
