"""Bounded, window-only Windows capture. No input injection or model training.

PrintWindow runs in a disposable process because it is a synchronous Windows API.
An image is observation evidence, not proof of fresh rendering or usable labels.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone

_DPI_READY = False


def _png(width: int, height: int, bgra: bytes, *, compression_level: int = 6) -> bytes:
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = bytearray()
    for y in range(height):
        row = bgra[y * width * 4:(y + 1) * width * 4]
        rgb = bytearray(width * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = row[2::4], row[1::4], row[0::4]
        rows.extend(b"\0" + rgb)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, level=compression_level)) + chunk(b"IEND", b""))


def region_digest(path: Path, region: tuple[int, int, int, int]) -> str:
    """Hash a region of our own RGB/filter-zero diagnostic PNG format."""
    raw, width, height = _diagnostic_rows(path)
    left, top, right, bottom = region
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("Invalid image region")
    digest = hashlib.sha256()
    stride = width * 3 + 1
    for y in range(top, bottom):
        digest.update(raw[y * stride + 1 + left * 3:y * stride + 1 + right * 3])
    return digest.hexdigest()


def _diagnostic_rows(path: Path) -> tuple[bytes, int, int]:
    """Decode only this module's bounded, filter-zero PNG capture format."""
    png = path.read_bytes()
    if png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ValueError("Not a diagnostic PNG")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", png[16:29])
    if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("Unsupported diagnostic PNG format")
    if not 1 <= width <= 7680 or not 1 <= height <= 4320:
        raise ValueError("Unsupported diagnostic PNG dimensions")
    offset, payload = 8, bytearray()
    while offset + 12 <= len(png):
        length = struct.unpack(">I", png[offset:offset + 4])[0]
        if offset + length + 12 > len(png):
            raise ValueError("Truncated PNG")
        if png[offset + 4:offset + 8] == b"IDAT":
            payload.extend(png[offset + 8:offset + 8 + length])
        offset += length + 12
    expected = (width * 3 + 1) * height
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, expected + 1)
    if len(raw) != expected or not decoder.eof:
        raise ValueError("Incorrect diagnostic PNG size")
    stride = width * 3 + 1
    for y in range(height):
        if raw[y * stride] != 0:
            raise ValueError("Only diagnostic unfiltered PNG rows are supported")
    return raw, width, height


def read_diagnostic_png(path: Path, *, stride: int = 1) -> tuple[bytes, int, int]:
    """Return sampled top-down BGRA from a saved capture without altering it."""
    if type(stride) is not int or not 1 <= stride <= 32:
        raise ValueError("Invalid sample stride")
    raw, width, height = _diagnostic_rows(path)
    out_width, out_height = (width + stride - 1) // stride, (height + stride - 1) // stride
    pixels = bytearray(out_width * out_height * 4)
    for out_y, y in enumerate(range(0, height, stride)):
        row = raw[y * (width * 3 + 1) + 1:(y + 1) * (width * 3 + 1)]
        target = out_y * out_width * 4
        for dst, source in ((0, 2), (1, 1), (2, 0)):
            pixels[target + dst:target + out_width * 4:4] = row[source::stride * 3]
    return bytes(pixels), out_width, out_height


def _sample_bgra(raw: bytes, width: int, height: int, stride: int) -> tuple[bytes, int, int]:
    if type(stride) is not int or stride < 1:
        raise ValueError('Sampling stride must be a positive integer')
    if stride == 1:
        return raw, width, height  # Preserve exact original bytes without an 8 MB channel copy.
    sampled_width = (width + stride - 1) // stride
    sampled_height = (height + stride - 1) // stride
    sampled = bytearray(sampled_width * sampled_height * 4)
    for out_y, source_y in enumerate(range(0, height, stride)):
        row = raw[source_y * width * 4:(source_y + 1) * width * 4]
        for channel in range(4):
            sampled[out_y * sampled_width * 4 + channel:(out_y + 1) * sampled_width * 4:4] = row[channel::stride * 4]
    return bytes(sampled), sampled_width, sampled_height


def _capture_worker(expected_exe: Path, *, pixels_only: bool = False, sample_stride: int = 1,
                    backend: str = "printwindow") -> dict:
    global _DPI_READY
    processing_started = time.perf_counter_ns()
    if os.name != "nt":
        raise OSError("Brotato window capture requires Windows")
    from ctypes import wintypes as w

    user = ctypes.WinDLL("user32", use_last_error=True)
    gdi = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)

    def api(dll, name, args, result):
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = args, result
        return fn

    enum = api(user, "EnumWindows", [callback_type, w.LPARAM], w.BOOL)
    visible = api(user, "IsWindowVisible", [w.HWND], w.BOOL)
    iconic = api(user, "IsIconic", [w.HWND], w.BOOL)
    window_pid = api(user, "GetWindowThreadProcessId", [w.HWND, ctypes.POINTER(w.DWORD)], w.DWORD)
    client_rect = api(user, "GetClientRect", [w.HWND, ctypes.POINTER(w.RECT)], w.BOOL)
    foreground = api(user, "GetForegroundWindow", [], w.HWND)
    open_process = api(kernel, "OpenProcess", [w.DWORD, w.BOOL, w.DWORD], w.HANDLE)
    close_handle = api(kernel, "CloseHandle", [w.HANDLE], w.BOOL)
    image_name = api(kernel, "QueryFullProcessImageNameW", [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)], w.BOOL)
    get_dc = api(user, "GetDC", [w.HWND], w.HDC)
    release_dc = api(user, "ReleaseDC", [w.HWND, w.HDC], ctypes.c_int)
    create_dc = api(gdi, "CreateCompatibleDC", [w.HDC], w.HDC)
    delete_dc = api(gdi, "DeleteDC", [w.HDC], w.BOOL)
    create_bitmap = api(gdi, "CreateCompatibleBitmap", [w.HDC, ctypes.c_int, ctypes.c_int], w.HBITMAP)
    select = api(gdi, "SelectObject", [w.HDC, w.HANDLE], w.HANDLE)
    delete_object = api(gdi, "DeleteObject", [w.HANDLE], w.BOOL)
    print_window = api(user, "PrintWindow", [w.HWND, w.HDC, w.UINT], w.BOOL)
    bit_blt = api(gdi, "BitBlt", [w.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                w.HDC, ctypes.c_int, ctypes.c_int, w.DWORD], w.BOOL)
    client_to_screen = api(user, "ClientToScreen", [w.HWND, ctypes.POINTER(w.POINT)], w.BOOL)
    window_at = api(user, "WindowFromPoint", [w.POINT], w.HWND)
    ancestor = api(user, "GetAncestor", [w.HWND, w.UINT], w.HWND)
    get_bits = api(gdi, "GetDIBits", [w.HDC, w.HBITMAP, w.UINT, w.UINT, ctypes.c_void_p, ctypes.c_void_p, w.UINT], ctypes.c_int)
    dpi = api(user, "SetProcessDpiAwarenessContext", [w.HANDLE], w.BOOL)
    if not _DPI_READY:
        if not dpi(ctypes.c_void_p(-4)):
            error = ctypes.get_last_error()
            current_dpi = api(user, "GetThreadDpiAwarenessContext", [], w.HANDLE)
            equal_dpi = api(user, "AreDpiAwarenessContextsEqual", [w.HANDLE, w.HANDLE], w.BOOL)
            if not equal_dpi(current_dpi(), ctypes.c_void_p(-4)):
                raise ctypes.WinError(error)
        _DPI_READY = True

    expected = os.path.normcase(str(expected_exe.resolve()))

    def identify(hwnd):
        pid = w.DWORD()
        if not window_pid(hwnd, ctypes.byref(pid)):
            return None
        handle = open_process(0x1000, False, pid.value)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            size = w.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not image_name(handle, 0, buffer, ctypes.byref(size)):
                return None
            return pid.value, os.path.normcase(str(Path(buffer.value).resolve()))
        finally:
            close_handle(handle)

    matches = []

    @callback_type
    def visit(hwnd, _):
        if visible(hwnd):
            identity = identify(hwnd)
            if identity and identity[1] == expected:
                matches.append((hwnd, identity))
        return True

    if not enum(visit, 0):
        raise OSError("Window enumeration failed")
    if len(matches) != 1:
        raise OSError(f"Expected one visible Brotato window; found {len(matches)}")
    hwnd, identity = matches[0]
    if iconic(hwnd):
        raise OSError("Brotato window is minimized")
    rect = w.RECT()
    if not client_rect(hwnd, ctypes.byref(rect)):
        raise OSError("Could not read Brotato client bounds")
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if not (1 <= width <= 7680 and 1 <= height <= 4320):
        raise OSError("Unsupported or empty Brotato client dimensions")
    if backend not in ("printwindow", "visible_client"):
        raise ValueError("Unknown capture backend")
    origin = w.POINT(0, 0)
    if backend == "visible_client":
        if foreground() != hwnd or not client_to_screen(hwnd, ctypes.byref(origin)):
            raise OSError("Visible-client capture requires foreground Brotato")
        for x in (1, width // 2, width - 2):
            for y in (1, height // 2, height - 2):
                if ancestor(window_at(w.POINT(origin.x + x, origin.y + y)), 2) != hwnd:
                    raise OSError("Brotato client region is covered by another window")
    dc_target = None if backend == "visible_client" else hwnd
    dc = get_dc(dc_target)
    memory = bitmap = old = None
    try:
        if not dc:
            raise OSError("Could not get Brotato device context")
        memory = create_dc(dc)
        bitmap = create_bitmap(dc, width, height)
        if not memory or not bitmap:
            raise OSError("Could not allocate capture buffer")
        old = select(memory, bitmap)
        if not old or old == ctypes.c_void_p(-1).value:
            old = None
            raise OSError("Could not select capture buffer")
        started = time.perf_counter_ns()
        was_foreground = foreground() == hwnd
        if backend == "visible_client":
            if not bit_blt(memory, 0, 0, width, height, dc, origin.x, origin.y, 0x00CC0020):
                raise OSError("Visible-client BitBlt failed")
        elif not print_window(hwnd, memory, 3):  # PW_CLIENTONLY | PW_RENDERFULLCONTENT
            raise OSError("PrintWindow failed; no desktop fallback attempted")
        finished = time.perf_counter_ns()
        after = w.RECT()
        if (identify(hwnd) != identity or iconic(hwnd) or not visible(hwnd)
                or not client_rect(hwnd, ctypes.byref(after))
                or (after.right - after.left, after.bottom - after.top) != (width, height)):
            raise OSError("Brotato identity or client bounds changed during capture")
        if backend == "visible_client" and foreground() != hwnd:
            raise OSError("Brotato lost foreground during visible-client capture")
        identity_verified = time.perf_counter_ns()
        select(memory, old)
        old = None
        # BITMAPINFOHEADER, top-down 32-bit BI_RGB bitmap.
        info = ctypes.create_string_buffer(struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 32, 0, width * height * 4, 0, 0, 0, 0))
        pixels = ctypes.create_string_buffer(width * height * 4)
        readback_started = time.perf_counter_ns()
        if get_bits(dc, bitmap, 0, height, pixels, info, 0) != height:
            raise OSError("Could not read complete capture bitmap")
        readback_finished = time.perf_counter_ns()
        raw = pixels.raw
        # Check every RGB pixel; alpha bytes from GDI are not meaningful.
        uniform = all(raw[i::4].count(raw[i]) == width * height for i in range(3))
        if uniform:
            raise OSError("Captured image is uniform; rendering could not be verified")
        report = {
            "pid": identity[0], "hwnd": hwnd, "executable": str(expected_exe.resolve()),
            "width": width, "height": height,
            "backend": "win32_visible_client_bitblt" if backend == "visible_client" else "win32_printwindow_client",
            "foreground_at_start": was_foreground, "foreground_at_finish": foreground() == hwnd,
            "capture_started_at_ns": started, "capture_finished_at_ns": finished,
            "capture_processing_started_at_ns": processing_started,
            "identity_verified_at_ns": identity_verified,
            "readback_started_at_ns": readback_started, "readback_finished_at_ns": readback_finished,
            "pixels_validated_at_ns": time.perf_counter_ns(),
            "rendered_at_ns": None, "fresh_render_verified": False,
            "clock": "perf_counter_ns_same_host",
        }
        if pixels_only:
            sampled, sampled_width, sampled_height = _sample_bgra(raw, width, height, sample_stride)
            return {**report, "pixels": sampled, "sample_width": sampled_width, "sample_height": sampled_height,
                    "sample_finished_at_ns": time.perf_counter_ns()}
        png = _png(width, height, raw)
        return {**report, "pixels_sha256": hashlib.sha256(raw).hexdigest(), "png_base64": base64.b64encode(png).decode("ascii")}
    finally:
        if old and memory:
            select(memory, old)
        if bitmap:
            delete_object(bitmap)
        if memory:
            delete_dc(memory)
        if dc:
            release_dc(dc_target, dc)


def capture_session(expected_exe: Path, output_root: Path, *, timeout: float = 10, backend: str = "printwindow") -> dict:
    """Capture one frame in a bounded child, then preserve a review-only session."""
    if expected_exe.name.lower() != "brotato.exe" or not expected_exe.is_file():
        raise OSError("Expected an installed Brotato.exe path")
    try:
        result = subprocess.run(
            [sys.executable, "-m", __name__, "--worker-exe", str(expected_exe.resolve()), "--backend", backend],
            capture_output=True, text=True, encoding="utf-8", timeout=timeout, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise OSError("Brotato capture timed out; capture worker terminated") from error
    if result.returncode:
        raise OSError(result.stderr.strip() or "Capture worker failed")
    report = json.loads(result.stdout)
    available = time.perf_counter_ns()
    png = base64.b64decode(report.pop("png_base64"), validate=True)
    session = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    (session / "frame.png").write_bytes(png)
    report.update({
        "session_id": session.name, "frame": "frame.png", "frame_sha256": hashlib.sha256(png).hexdigest(),
        "available_at_ns": available, "clock": "perf_counter_ns_same_host",
        "status": "captured_pending_visual_review", "usable_for_training": False,
        "training_performed": False, "runtime_ready": False,
        "missing": ["verified_screen_state", "action_and_application_labels", "outcome", "trainer"],
    })
    (session / "observation.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {**report, "session_directory": str(session.resolve())}


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-exe", type=Path, required=True)
    parser.add_argument("--backend", choices=("printwindow", "visible_client"), default="printwindow")
    args = parser.parse_args()
    try:
        print(json.dumps(_capture_worker(args.worker_exe, backend=args.backend), ensure_ascii=True))
        return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
