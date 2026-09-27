"""Separate hidden process finalizes an owned recording if its session process dies."""

if __name__ == "__main__":
    from playmodel.execution_log import launch_module
    launch_module('playmodel.video_guard')

import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import time

from .obs import ObsClient
from .video import SCENE


def guard(directory: Path, parent_pid: int, seconds: float):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, parent_pid)
    deadline = time.monotonic() + seconds
    try:
        while True:
            report = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
            if report.get('closed'):
                return
            dead = not handle or kernel.WaitForSingleObject(handle, 500) == 0
            if not dead and time.monotonic() < deadline:
                continue
            with ObsClient() as obs:
                status = obs.call('GetRecordStatus')
                elapsed_ms = (time.perf_counter_ns()-report['timeline_origin_ns'])/1e6
                if (status['outputActive'] and not status['outputPaused']
                        and abs(elapsed_ms-status['outputDuration']) < 2500
                        and obs.call('GetCurrentProgramScene')['currentProgramSceneName'] == SCENE):
                    report['output_path'] = obs.call('StopRecord')['outputPath']
                    for _ in range(100):
                        if not obs.call('GetRecordStatus')['outputActive']:
                            break
                        time.sleep(.1)
                    for name, value in report.get('previous_audio_mutes', {}).items():
                        if obs.call('GetInputMute', inputName=name)['inputMuted']:
                            obs.call('SetInputMute', inputName=name, inputMuted=value)
                    target = (directory / 'original').resolve()
                    if Path(obs.call('GetRecordDirectory')['recordDirectory']).resolve() == target:
                        obs.call('SetRecordDirectory', recordDirectory=report['previous_record_directory'])
                    report['guardian_recovery'] = 'parent_exit' if dead else 'hard_deadline'
                    report['closed'] = True
                    (directory / 'guardian-recovery.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                    if dead:
                        (directory / 'recording.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                return
    finally:
        if handle:
            kernel.CloseHandle(handle)


if __name__ == '__main__':
    guard(Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))
