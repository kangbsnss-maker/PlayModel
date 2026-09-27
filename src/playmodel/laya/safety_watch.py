"""Read-only incident observation. This module has no input writer or rearm path."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import threading


def user_stop(error):
    text = str(error).casefold()
    return any(term in text for term in ('f8', 'human intervention', 'human_intervention',
                                        'user stop', 'user_stop', 'session stopped by user'))


def watch_ui(executable, directory, *, stop_file, status, incident,
             capture_factory=None, ocr_factory=None, stopped=None, sleep=time.sleep):
    """Keep observing uncertain transport/identity faults without concealing them.

    Known released visual boundaries recover inside the session. An unclassified
    fault has no rearm proof: even two readable frames cannot authorize input.
    """
    from playmodel.games.brotato.menu_capture import MenuCapture
    from playmodel.games.brotato.ocr import MenuOcr
    from playmodel.games.brotato.menu import classify_scene
    from playmodel.games.brotato.ui_layers import UiLayers
    directory, stop_file = Path(directory), Path(stop_file)
    output = directory / ('safety-observation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'incident.json').write_text(json.dumps(incident, ensure_ascii=False, indent=2), encoding='utf8')
    latch_stop, f8_latched = threading.Event(), threading.Event()
    watcher = None
    if stopped is None:
        import ctypes
        def watch_stop():
            while not latch_stop.wait(.01):
                if ctypes.windll.user32.GetAsyncKeyState(0x77) & 0x8000:
                    f8_latched.set()
                    return
        watcher = threading.Thread(target=watch_stop, name='read-only-F8-latch', daemon=True)
        watcher.start()
        stopped = f8_latched.is_set
    def check():
        if stop_file.exists() or stopped():
            raise InterruptedError('User stopped observation')
    layers = UiLayers()
    capture = (capture_factory or (lambda: MenuCapture(executable, fps=2)))()
    ocr = (ocr_factory or (lambda: MenuOcr(Path(__file__).resolve().parents[3] / 'scripts/windows_ocr.ps1')))()
    attempts, failures = 0, 0
    try:
        while True:
            check()
            attempts += 1
            try:
                shot, pixels, width, height = capture.read(output / 'frames', check=check, timeout=4)
                source = Path(shot['session_directory']) / 'frame.png'
                result = ocr.read(source)
                check()
                scene = classify_scene(result, width=width, height=height).scene
                layer = layers.observe(scene, frame_ref=source, observed_at_ns=shot['capture_started_at_ns'])
                with (output / 'ui-layers.jsonl').open('a', encoding='utf8') as stream:
                    stream.write(json.dumps(layer, ensure_ascii=False) + '\n')
                status.update(status='observing', phase='waiting_safety',
                    input_authorized=False, automatic_rearm_allowed=False,
                    learning_quarantined=True, recovery_incident=incident, ui_layers=layer,
                    recovery_observations=attempts, recovery_capture_errors=failures)
            except InterruptedError:
                raise
            except (OSError, ValueError, KeyError) as error:
                failures += 1
                capture.close()
                status.update(status='observing', phase='waiting_safety',
                    input_authorized=False, automatic_rearm_allowed=False,
                    learning_quarantined=True, recovery_incident=incident,
                    recovery_observations=attempts, recovery_capture_errors=failures,
                    recovery_observation_error=f'{type(error).__name__}: {error}')
            # Check STOP/F8 during backoff. No tight relaunch or input loop.
            for _ in range(20):
                check()
                sleep(.25)
    except InterruptedError:
        if f8_latched.is_set() and not stop_file.exists():
            stop_file.write_text('User F8 during observation recovery', encoding='utf8')
        status.update(status='user_stopped', phase='stopped', stop_category='user_stop')
        return 0
    finally:
        latch_stop.set()
        if watcher is not None:
            watcher.join(.1)
        capture.close()
        ocr.close()
