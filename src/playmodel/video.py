"""Full-session OBS capture and evidence-based, removable narration captions."""
from __future__ import annotations

import json
import base64
import html as html_module
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from .obs import ObsClient

SCENE = 'PlayModel - Brotato'
SOURCE = 'PlayModel - Brotato Game'


class SessionRecording:
    """Own only recordings started here. Keep network/disk work off movement path."""
    def __init__(self, directory: Path, *, max_seconds: float):
        self.directory = directory
        self.deadline = time.monotonic() + max_seconds
        self.events = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.error = None
        self.report = {'scope': 'full_session', 'subtitle_burned_in': False}
        self._thread = threading.Thread(target=self._run, daemon=True, name='obs-session-recording')
        self._thread.start()
        if not self._ready.wait(25):
            self._stop.set()
            raise OSError('OBS recording startup timeout')
        if self.error:
            raise OSError(self.error)

    def event(self, kind: str, text: str, **details):
        with self._lock:
            self.events.append({'at_ns': time.perf_counter_ns(), 'kind': kind,
                                'text': text, 'details': details})

    def _persist(self):
        with self._lock:
            report = {**self.report, 'events': list(self.events)}
        temporary = self.directory / 'recording.tmp.json'
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        for attempt in range(10):
            try:
                os.replace(temporary, self.directory / 'recording.json')
                break
            except PermissionError:
                # Windows readers/antivirus can briefly hold a non-delete-sharing
                # handle. Retry the atomic replace without truncating the old file.
                if attempt == 9:
                    raise
                time.sleep(.05)

    def _run(self):
        owned = False
        muted = {}
        old_directory = None
        target_directory = str((self.directory / 'original').resolve())
        self.directory.mkdir(parents=True, exist_ok=True)
        Path(target_directory).mkdir(parents=True, exist_ok=True)
        obs = None
        try:
            obs = ObsClient()
            if obs.call('GetRecordStatus')['outputActive'] or obs.call('GetStreamStatus')['outputActive']:
                raise OSError('Existing OBS output active; recording ownership refused')
            if obs.call('GetCurrentProgramScene')['currentProgramSceneName'] != SCENE:
                raise OSError('Select the calibrated PlayModel - Brotato scene')
            items = obs.call('GetSceneItemList', sceneName=SCENE)['sceneItems']
            enabled = [item for item in items if item['sceneItemEnabled']]
            if len(enabled) != 1 or enabled[0]['sourceName'] != SOURCE or enabled[0]['inputKind'] != 'game_capture':
                raise OSError('Recording scene must contain only the Brotato game capture')
            settings = obs.call('GetInputSettings', inputName=SOURCE)['inputSettings']
            if settings.get('capture_mode') != 'window' or settings.get('window') != 'Brotato:Engine:Brotato.exe':
                raise OSError('Brotato capture target changed')
            # Requesting a screenshot also rejects a not-yet-hooked capture source.
            obs.call('GetSourceScreenshot', sourceName=SOURCE, imageFormat='png', imageWidth=320)
            # Narration is recorded later; avoid desktop notification audio and live microphone.
            for item in obs.call('GetInputList')['inputs']:
                if item['inputKind'] in ('wasapi_input_capture', 'wasapi_output_capture'):
                    name = item['inputName']
                    muted[name] = obs.call('GetInputMute', inputName=name)['inputMuted']
                    obs.call('SetInputMute', inputName=name, inputMuted=True)
            obs.call('SetInputMute', inputName=SOURCE, inputMuted=False)
            old_directory = obs.call('GetRecordDirectory')['recordDirectory']
            self.report['previous_record_directory'] = old_directory
            obs.call('SetRecordDirectory', recordDirectory=target_directory)
            self.report['start_request_ns'] = time.perf_counter_ns()
            obs.call('StartRecord')
            owned = True
            startup_deadline = time.monotonic() + 10
            status = obs.call('GetRecordStatus')
            while not status['outputActive'] and time.monotonic() < startup_deadline:
                time.sleep(.1)
                status = obs.call('GetRecordStatus')
            observed_ns = time.perf_counter_ns()
            if not status['outputActive'] or status['outputPaused']:
                raise OSError('OBS did not enter active unpaused recording')
            self.report['timeline_origin_ns'] = observed_ns - round(status['outputDuration'] * 1e6)
            self.report['timing_uncertainty_ms'] = (observed_ns - self.report['start_request_ns']) / 1e6
            self.report['started'] = True
            self.report['previous_audio_mutes'] = muted
            self._persist()
            subprocess.Popen([sys.executable, '-m', 'playmodel.video_guard', str(self.directory.resolve()),
                              str(os.getpid()), str(max(1, self.deadline-time.monotonic()+15))],
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._ready.set()
            while not self._stop.wait(.5):
                if time.monotonic() >= self.deadline:
                    raise OSError('Recording hard time limit')
                status = obs.call('GetRecordStatus')
                if not status['outputActive'] or status['outputPaused']:
                    # A user stopped/paused: do not automatically restart or resume.
                    raise OSError('OBS recording stopped or paused externally')
                if obs.call('GetCurrentProgramScene')['currentProgramSceneName'] != SCENE:
                    raise OSError('OBS scene changed externally')
                self._persist()
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            self.report['error'] = self.error
        finally:
            if owned and obs is not None:
                try:
                    if obs.call('GetRecordStatus')['outputActive']:
                        self.report['output_path'] = obs.call('StopRecord')['outputPath']
                        stop_deadline = time.monotonic() + 10
                        while obs.call('GetRecordStatus')['outputActive']:
                            if time.monotonic() >= stop_deadline:
                                raise OSError('OBS output finalization timeout')
                            time.sleep(.1)
                    else:
                        self.report['output_path_unavailable'] = True
                except Exception as error:
                    self.report['stop_error'] = str(error)
            if obs is not None:
                for name, value in muted.items():
                    try:
                        if obs.call('GetInputMute', inputName=name)['inputMuted']:
                            obs.call('SetInputMute', inputName=name, inputMuted=value)
                    except Exception:
                        self.report['audio_restore_failed'] = True
                if old_directory is not None:
                    try:
                        if obs.call('GetRecordDirectory')['recordDirectory'] == target_directory:
                            obs.call('SetRecordDirectory', recordDirectory=old_directory)
                    except Exception:
                        self.report['directory_restore_failed'] = True
                obs.close()
            self.report['closed'] = 'stop_error' not in self.report
            self._persist()
            self._ready.set()

    def close(self) -> dict:
        self._stop.set()
        self._thread.join(45)
        if self._thread.is_alive():
            raise OSError('OBS recording shutdown incomplete')
        with self._lock:
            self.report['events'] = list(self.events)
        (self.directory / 'recording.json').write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding='utf-8')
        return self.report


def run_media(args: list[str], *, timeout=180) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                            | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
    if result.returncode:
        raise OSError(f'Media command failed: {result.stderr[-2000:]}')
    return result


def timestamp(seconds: float) -> str:
    total = max(0, round(seconds * 1000))
    hours, rest = divmod(total, 3600000)
    minutes, rest = divmod(rest, 60000)
    sec, ms = divmod(rest, 1000)
    return f'{hours:02}:{minutes:02}:{sec:02},{ms:03}'


def edit_plan(events: list[dict], origin_ns: int, duration: float) -> tuple[list[dict], list[tuple[float, float]]]:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Positive video duration required')
    captions = []
    windows = [(0.0, min(duration, 6.0))]
    for event in sorted(events, key=lambda event: event['at_ns']):
        start = max(0., (event['at_ns'] - origin_ns) / 1e9)
        if start >= duration or not event['text'].strip():
            continue
        captions.append({'start': start, 'end': min(duration, start + 4.0),
                         'text': event['text'], 'kind': event['kind']})
        if event['kind'] in ('combat_start', 'wave_result', 'training_update', 'menu_choice', 'shop_reroll', 'stop'):
            windows.append((max(0., start - 2), min(duration, start + (10 if event['kind'] == 'combat_start' else 5))))
    # No overlapping narration cues. Repeated same-state frame callbacks do not create events.
    for index, caption in enumerate(captions[:-1]):
        caption['end'] = min(caption['end'], captions[index + 1]['start'])
    captions = [cue for cue in captions if cue['end'] - cue['start'] >= 1.0]
    merged = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + .5:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return captions, merged


def write_srt(path: Path, captions: list[dict]):
    path.write_text('\n\n'.join(f"{i}\n{timestamp(c['start'])} --> {timestamp(c['end'])}\n{c['text']}"
                               for i, c in enumerate(captions, 1)) + '\n', encoding='utf-8-sig')


def write_preview(directory: Path, captions: list[dict], *, video_name: str = 'highlights.mp4'):
    vtt = 'WEBVTT\n\n' + '\n\n'.join(
        f"{timestamp(c['start']).replace(',', '.')} --> {timestamp(c['end']).replace(',', '.')}\n{c['text']}"
        for c in captions) + '\n'
    (directory / 'highlights.ko.vtt').write_text(vtt, encoding='utf-8')
    encoded = base64.b64encode(vtt.encode()).decode()
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Brotato 녹음용 미리보기</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 20px;background:#17191d;color:#f2f2f2;font:17px/1.7 sans-serif}video{width:100%;max-height:78vh}a{color:#91c4ff}label{cursor:pointer}video::cue{font-size:22px;background:#000c;color:white}</style>
<p><strong>Brotato 학습 · 녹음용 미리보기</strong></p>
<video id="video" controls preload="metadata" src="highlights.mp4"><track id="captions" kind="subtitles" srclang="ko" label="상황 설명" default></video>
<p><label><input id="toggle" type="checkbox" checked> 설명 자막 표시</label> · <a href="narration.ko.md">녹음 대본</a> · <a href="highlights.ko.srt" download>SRT 다운로드</a></p>
<p>자막은 영상에 포함되지 않았습니다. 녹음 후 자막 트랙을 끄거나 제거하세요. 실패·중단도 이번 실험 기록입니다.</p>
<script>const bytes=Uint8Array.from(atob(''' + json.dumps(encoded) + '''),c=>c.charCodeAt(0));
const track=document.getElementById('captions');track.src=URL.createObjectURL(new Blob([bytes],{type:'text/vtt'}));
function mode(){track.track.mode=document.getElementById('toggle').checked?'showing':'disabled';}
track.addEventListener('load',mode);document.getElementById('toggle').addEventListener('change',mode);mode();</script></html>'''
    html=html.replace('src="highlights.mp4"','src="'+html_module.escape(video_name,quote=True)+'"')
    (directory / 'preview.html').write_text(html, encoding='utf-8')


def make_highlights(directory: Path, *, output_directory: Path | None = None) -> dict:
    """Keep original intact; create clean MP4 + external Korean SRT + narration notes."""
    report = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
    directory = output_directory or directory
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(report['output_path'])
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise OSError('Local FFmpeg and ffprobe required for editing')
    for attempt in range(20):
        try:
            probe = json.loads(run_media([ffprobe, '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(source)]).stdout)
            break
        except OSError:
            if attempt == 19:
                raise
            time.sleep(.5)
    duration = float(probe['format']['duration'])
    events = []
    for original in report['events']:
        event = dict(original)
        concise = {'session_start': '로컬 학습 시작 · 전 과정 녹화',
                   'combat_start': '로컬 이동 정책으로 아이템 수집·위험 회피 시도',
                   'training_update': '이동 가중치 갱신 완료 · 성능 향상 미검증',
                   'wave_result': '전투 제어 종료 · 결과 화면 확인',
                   'stop': '학습 종료 · 원본 보존 및 편집본 생성'}
        if event['kind'] in concise:
            event['text'] = concise[event['kind']]
        if event['kind'] == 'stop' and event.get('details', {}).get('reason') != 'wave_limit':
            event['text'] = '인식·제어 확인이 중단되어 자동 정지\n확인되지 않은 구간은 학습에서 제외'
        events.append(event)
    captions, intervals = edit_plan(events, report['timeline_origin_ns'], duration)
    # Final still gives the user time to read the stop reason and record narration.
    end_card_seconds = 5.
    write_srt(directory / 'full-session.ko.srt', captions)
    if captions and captions[-1]['kind'] == 'stop':
        captions[-1]['end'] = duration + end_card_seconds
    segments, highlight_captions = [], []
    offset = 0.
    for index, (start, end) in enumerate(intervals):
        target = directory / f'cut-{index:03}.mp4'
        if target.exists():
            raise FileExistsError(target)
        padding = end_card_seconds if index == len(intervals)-1 and end == duration else 0.
        filters = ['-vf', f'tpad=stop_mode=clone:stop_duration={padding}'] if padding else []
        if padding and any(s['codec_type'] == 'audio' for s in probe['streams']):
            filters += ['-af', f'apad=pad_dur={padding}']
        run_media([ffmpeg, '-nostdin', '-v', 'error', '-n', '-ss', str(start), '-i', str(source), '-t', str(end-start+padding),
                   '-map', '0:v:0', '-map', '0:a?', *filters, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                   '-threads', '2', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', str(target)])
        segments.append(target)
        for cue in captions:
            left, right = max(start, cue['start']), min(end+padding, cue['end'])
            if right - left >= .25:
                highlight_captions.append({**cue, 'start': offset + left-start, 'end': offset+right-start})
        offset += end-start+padding
    concat = directory / 'cuts.ffconcat'
    # Only our generated basenames, no input path escaping/interpolation into shell code.
    concat.write_text('ffconcat version 1.0\n' + ''.join(f"file '{p.name}'\n" for p in segments), encoding='utf-8')
    target = directory / ((report['video_title']+'__Highlights.mp4') if report.get('video_title') else 'highlights.mp4')
    run_media([ffmpeg, '-nostdin', '-v', 'error', '-n', '-f', 'concat', '-safe', '1', '-i', str(concat),
               '-c', 'copy', '-movflags', '+faststart', str(target)])
    write_srt(directory / 'highlights.ko.srt', highlight_captions)
    write_preview(directory, highlight_captions, video_name=target.name)
    (directory / 'narration.ko.md').write_text('# 녹음용 상황 설명\n\n'
        'SRT는 별도 파일입니다. 녹음 후 자막 트랙을 비활성화하거나 제거하면 됩니다. 영상에는 자막을 굽지 않았습니다.\n\n'
        + '\n'.join(f"- {timestamp(c['start'])} — {c['text']}" for c in highlight_captions), encoding='utf-8')
    result = {'original': str(source), 'highlights': str(target), 'original_seconds': duration,
              'highlight_seconds_planned': offset, 'cuts': intervals, 'captions': len(highlight_captions),
              'selection': 'event_rules_not_semantic_ai', 'subtitle_burned_in': False,
              'source_probe': probe}
    (directory / 'edit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result
