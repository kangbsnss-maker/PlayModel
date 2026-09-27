"""Local desktop launcher; learning workers outlive this window and use no LLM API."""
from __future__ import annotations

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from playmodel.execution_log import event, exception

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/local/learner-launch.json'
RUNTIME = ROOT / 'artifacts/local-learning'
STOP = ROOT / 'artifacts/BROTATO_STOP'
LOCK = RUNTIME / 'worker.lock'
PHASES = {'initializing': '준비', 'obs_start': 'OBS 실행·연결', 'cycle_start': '다음 학습 준비',
          'laya_model_preload': 'Laya 판단 모델 준비', 'laya_training': 'Laya 판단 가중치 학습',
          'new_run_setup': '캐릭터·무기·난이도 선택', 'combat': '전투 경험 수집',
          'menu': '성장·상점 선택', 'training': '가중치 학습',
          'training_worker_started': '별도 학습 작업 준비',
          'await_training_worker': '판 종료 · 학습 결과 대기',
          'comparison_complete': '새 판 비교 완료', 'stopped': '중지',
          'recovery': '진행 중인 판 복구', 'candidate_rejected': '후보 검사 탈락',
          'collect_training': '학습용 새 판 수집', 'training_collected': '학습 판 저장 완료',
          'train_candidate': '후보 모델 학습', 'candidate_saved': '후보 모델 저장 완료',
          'collect_evaluation': '독립 평가 판 수집', 'evaluation_collected': '평가 판 저장 완료',
          'bounded_observation_recovery': '화면 재확인', 'active_run_recovery': '진행 중인 판 마무리',
          'awaiting_game_resume': '게임 재개 대기 · 입력 없이 관측 중',
          'waiting_observation': '화면 확인 대기 · 입력 없이 관측 중',
          'waiting_safety': '복구 증거 확인 · UI 관측 유지 · 조작 보류'}
STATES = {'observing': 'UI 관측 계속', 'fault_paused': '오류 기록 확인 필요', 'user_stopped': '사용자 중지',
          'budget_paused': '실행 예산 종료', 'budget_complete': '요청한 실행 완료',
          'candidate_rejected': '후보 검사 탈락', 'starting': '시작 준비'}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def format_learning_counts(counts):
    """Show accepted optimizer updates, not decisions mislabeled as learning."""
    keys = ('menu_decisions', 'tactic_decisions', 'accepted_updates', 'trained_decisions', 'runs_completed')
    if any(type(counts.get(key)) is not int or counts[key] < 0 for key in keys):
        raise ValueError('Verified learning counters required')
    return (f"누적 판단 {counts['menu_decisions'] + counts['tactic_decisions']:,}회   ·   "
            f"학습(가중치 갱신) {counts['accepted_updates']:,}회\n"
            f"유효 학습 자료 {counts['trained_decisions']:,}건   ·   완료 학습 {counts['runs_completed']:,}판")


class LearningCounterReader:
    """Read cached evidence off the Tk thread; no browser or learner is required."""
    def __init__(self, root, *, fetch=None, start=True):
        self.root = Path(root)
        self.fetch = fetch or self._fetch
        self.dashboard = None
        self.snapshot = None
        self.error = None
        self.stop = threading.Event()
        if start:
            threading.Thread(target=self._run, name='desktop-learning-counters', daemon=True).start()

    def _fetch(self):
        info = read_json(self.root / 'artifacts/local-learning/dashboard.json')
        url = info.get('url', '')
        try:
            parsed = urlsplit(url)
            if (parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.port
                    and not parsed.username and not parsed.password and not parsed.query
                    and not parsed.fragment and parsed.path in ('', '/')):
                with build_opener(ProxyHandler({})).open(url.rstrip('/') + '/api/status', timeout=1) as response:
                    data = json.loads(response.read(4_000_000))
                if data.get('schema') == 'playmodel.laya-dashboard.v1':
                    return data
        except (OSError, ValueError):
            pass
        if self.dashboard is None:
            from laya_dashboard import Dashboard
            self.dashboard = Dashboard(self.root)
        return self.dashboard.status()

    def refresh_once(self):
        try:
            from playmodel.games.brotato.focus_settings import focus_pause_status
            data = self.fetch()
            text = format_learning_counts(data['counts'])
            warnings = len(data.get('errors', []))
            self.snapshot = {'text': text, 'at': time.monotonic(), 'evidence_warnings': warnings,
                             'focus_settings': focus_pause_status()}
            self.error = None
        except Exception as error:
            # Keep the last verified values; a display fault cannot stop input.
            self.error = str(error)

    def _run(self):
        while not self.stop.is_set():
            self.refresh_once()
            self.stop.wait(3)


def worker_running() -> bool:
    """Probe the coordinator's OS lock, never infer life from a stale PID file."""
    import msvcrt
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+b') as stream:
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return True
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    return False


def latest_status() -> tuple[Path | None, dict]:
    paths = list((ROOT / 'artifacts/recurrent-cycles').glob('*/status.json'))
    paths += list((ROOT / 'artifacts/laya-learning').glob('*/status.json'))
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        value = read_json(path)
        if value:
            return path, value
    return None, {}


def ensure_video_titles():
    """Independent, local metadata worker. Existing worker exits via OS lock."""
    script = ROOT / 'scripts/label_learning_videos.py'
    if not script.is_file():
        return
    with (RUNTIME / 'video-titles.log').open('ab') as output:
        child = subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8',
                                  str(script), '--watch'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    event('child_started', child_pid=child.pid, program='label_learning_videos.py')


def open_dashboard(*, show_browser=True):
    """Reuse the loopback service; its lifetime is independent of learning."""
    def endpoint():
        info = read_json(RUNTIME / 'dashboard.json')
        url = info.get('url', '')
        try:
            parsed = urlsplit(url)
            if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
                    or parsed.username or parsed.password or parsed.path not in ('', '/')
                    or parsed.query or parsed.fragment or not parsed.port):
                return None
            with build_opener(ProxyHandler({})).open(url.rstrip('/') + '/api/health', timeout=.5) as response:
                health = json.loads(response.read(4096))
            if health.get('service') == 'playmodel-laya-dashboard' and health.get('pid') == info.get('pid'):
                return url
        except (OSError, ValueError):
            return None
        return None
    url = endpoint()
    if url is None:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        with (RUNTIME / 'dashboard.log').open('ab') as log:
            subprocess.Popen([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8',
                              str(ROOT / 'scripts/laya_dashboard.py')], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        until = time.monotonic() + 10
        while url is None and time.monotonic() < until:
            time.sleep(.1)
            url = endpoint()
        if url is None:
            raise RuntimeError('학습 대시보드를 열지 못했습니다. dashboard.log를 확인하세요.')
    if show_browser:
        os.startfile(url)
    return url


def launch_worker(checkpoint: Path, resume_summary: Path | None, *, resume_latest=True, backend=None) -> tuple[int, Path]:
    if worker_running():
        raise RuntimeError('학습 프로그램이 이미 실행 중입니다.')
    if not checkpoint.is_file():
        raise ValueError('모델 파일을 선택하세요.')
    if resume_summary and not resume_summary.is_file():
        raise ValueError('이어서 평가할 summary.json 파일이 없습니다.')
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.is_file():
        raise ValueError('.venv Python 환경이 없습니다. 실행 가이드를 확인하세요.')
    backend = backend or read_json(CONFIG).get('backend', 'ppo')
    if backend not in ('ppo', 'laya'):
        raise ValueError('알 수 없는 판단 모델입니다.')
    command = [str(python), '-X', 'utf8', str(ROOT / 'scripts/run_recurrent_cycle.py'),
               str(checkpoint.resolve()), '--continuous', '--recover-active-run', '--evaluation-runs', '1',
               '--max-run-seconds', '1800', '--device', 'cuda']
    if resume_latest:
        command += ['--resume-latest']
    if resume_summary:
        command += ['--resume-summary', str(resume_summary.resolve())]
    else:
        command += ['--online-updates']
    if backend == 'laya':
        model_dir = ROOT / 'models/laya/base'
        if not (model_dir / 'source-manifest.json').is_file() or not (ROOT / '.venv-laya/Scripts/python.exe').is_file():
            raise ValueError('로컬 Laya 설치가 필요합니다. docs/guides/laya.md를 확인하세요.')
        command = [str(python), '-X', 'utf8', str(ROOT / 'scripts/run_laya_learning.py'),
                   str(checkpoint.resolve()), '--model-dir', str(model_dir), '--continuous', '--campaign',
                   '--recover-active-run', '--max-run-seconds', '1800', '--device', 'cuda']
        if resume_latest:
            reports = (ROOT / 'artifacts/laya-learning').glob('*/laya/update-*/report.json')
            for report_path in sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True):
                report = read_json(report_path)
                candidate = report.get('checkpoint')
                if report.get('accepted') is True and candidate and Path(candidate).is_file():
                    command += ['--laya-checkpoint', candidate]
                    break
    # The user explicitly requested resume by pressing Start. Never clear this
    # file during background retries or passive application startup.
    STOP.unlink(missing_ok=True)
    log = RUNTIME / ('worker-' + time.strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns()) + '.log')
    with log.open('ab') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    event('child_started', child_pid=child.pid, program=Path(command[3]).name,
          output_log=str(log), checkpoint=str(checkpoint.resolve()))
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({'checkpoint': str(checkpoint.resolve()),
                                 'resume_summary': str(resume_summary.resolve()) if resume_summary else '',
                                 'resume_latest': True, 'backend': backend},
                                ensure_ascii=False, indent=2), encoding='utf-8')
    (RUNTIME / 'launch.json').write_text(json.dumps({'pid': child.pid, 'log': str(log),
        'command': command, 'started_at': time.time(), 'agent_or_api_required': False}, indent=2), encoding='utf-8')
    try:
        ensure_video_titles()
    except OSError as error:
        exception('video_title_worker_launch_failed', error)
    return child.pid, log


class App:
    def __init__(self, window: tk.Tk):
        self.window = window
        self.pending_until = 0.0
        window.title('PlayModel — 로컬 자체 학습')
        icon = ROOT / 'assets/app-icon/playmodel.ico'
        if icon.is_file() and sys.platform == 'win32':
            window.iconbitmap(default=str(icon))
        window.geometry('820x730')
        window.minsize(720, 690)
        frame = ttk.Frame(window, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='PlayModel', font=('Segoe UI', 23, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='게임 플레이 · 상황 판단 · 결과 기반 학습\n정상 실행에는 에이전트나 API 토큰이 필요하지 않습니다.').pack(anchor='w', pady=(4, 16))
        self.learning_counts = tk.StringVar(value='학습 통계 불러오는 중…')
        ttk.Label(frame, textvariable=self.learning_counts, font=('Segoe UI', 13, 'bold'),
                  justify='left', wraplength=760).pack(anchor='w', pady=(0, 4))
        self.counts_note = tk.StringVar(value='원본 학습 기록을 확인합니다.')
        ttk.Label(frame, textvariable=self.counts_note).pack(anchor='w', pady=(0, 12))
        self.counter_reader = LearningCounterReader(ROOT)
        config = read_json(CONFIG)
        self.backend = tk.StringVar(value=config.get('backend', 'ppo'))
        ttk.Label(frame, text='판단 모델: laya = 전투 전술·성장·상점 학습 / ppo = 기존 이동·선택 학습').pack(anchor='w')
        ttk.Combobox(frame, textvariable=self.backend, values=('laya', 'ppo'), state='readonly', width=12).pack(anchor='w')
        self.checkpoint = tk.StringVar(value=config.get('checkpoint', str(ROOT / 'models/recurrent/20260927-ppo-v1/ppo-candidate.pt')))
        self.resume = tk.StringVar(value=config.get('resume_summary', ''))
        self.resume_latest = tk.BooleanVar(value=config.get('resume_latest', True))
        self._path_row(frame, '시작 모델 (.pt)', self.checkpoint, [('PyTorch model', '*.pt')])
        self._path_row(frame, '중단된 비교 기록 (선택)', self.resume, [('Cycle summary', '*.json')])
        ttk.Checkbutton(frame, text='저장된 진행 이어서 (해제하면 새 실험)', variable=self.resume_latest).pack(anchor='w', pady=(8, 0))
        controls = ttk.Frame(frame)
        controls.pack(fill='x', pady=18)
        self.start_button = ttk.Button(controls, text='학습 시작 / 이어서', command=self.start)
        self.start_button.pack(side='left')
        ttk.Button(controls, text='안전 중지', command=self.stop).pack(side='left', padx=8)
        ttk.Button(controls, text='실행 가이드', command=lambda: os.startfile(ROOT / 'docs/guides/standalone-learning.html')).pack(side='left')
        ttk.Button(controls, text='기록 폴더', command=self.open_records).pack(side='left', padx=8)
        ttk.Button(controls, text='실행 로그', command=self.open_logs).pack(side='left')
        ttk.Button(controls, text='영상 제목', command=lambda: os.startfile(ROOT / 'media/learning-videos.html')).pack(side='left', padx=8)
        self.dashboard_button = ttk.Button(frame, text='학습 도표 · 사람 전술 선호도 · 판단 근거', command=self.open_dashboard)
        self.dashboard_button.pack(anchor='w', pady=(0, 12))
        self.state = tk.StringVar(value='실행 상태 확인 중')
        ttk.Label(frame, textvariable=self.state, font=('Segoe UI', 12, 'bold'), wraplength=720).pack(anchor='w')
        self.details = tk.StringVar()
        ttk.Label(frame, textvariable=self.details, wraplength=720, justify='left').pack(anchor='w', pady=10)
        ttk.Label(frame, text='창을 닫아도 학습은 계속됩니다. 중지는 위 버튼 또는 F8.\nUI 변화는 재관측합니다. 전송·해제 상태가 불명확하면 조작만 보류하고 관측을 유지합니다.',
                  wraplength=720).pack(side='bottom', anchor='w')
        self.refresh()

    def _path_row(self, parent, title, variable, types):
        ttk.Label(parent, text=title).pack(anchor='w', pady=(5, 2))
        row = ttk.Frame(parent)
        row.pack(fill='x')
        ttk.Entry(row, textvariable=variable).pack(side='left', fill='x', expand=True)
        def choose():
            path = filedialog.askopenfilename(initialdir=ROOT, filetypes=types)
            if path:
                variable.set(path)
        ttk.Button(row, text='찾기', command=choose).pack(side='left', padx=(8, 0))

    def open_dashboard(self):
        self.dashboard_button.configure(state='disabled')
        def work():
            try:
                open_dashboard()
            except Exception as error:
                exception('dashboard_launch_failed', error)
                message = str(error)
                self.window.after(0, lambda: messagebox.showerror('대시보드 오류', message))
            finally:
                self.window.after(0, lambda: self.dashboard_button.configure(state='normal'))
        threading.Thread(target=work, name='dashboard-launcher', daemon=True).start()

    def start(self):
        try:
            summary = Path(self.resume.get().strip()) if self.resume.get().strip() else None
            pid, _ = launch_worker(Path(self.checkpoint.get().strip()), summary,
                                   resume_latest=self.resume_latest.get(), backend=self.backend.get())
            self.resume_latest.set(True)
            self.pending_until = time.monotonic() + 15
            self.state.set(f'시작 중 · 프로세스 {pid}')
            self.start_button.state(['disabled'])
        except Exception as error:
            exception('launch_failed', error)
            messagebox.showerror('시작 실패', str(error))

    def stop(self):
        event('user_stop_requested')
        STOP.parent.mkdir(parents=True, exist_ok=True)
        STOP.write_text('Explicit stop from local desktop application\n', encoding='utf-8')
        self.state.set('중지 요청됨 · 입력 해제와 기록 저장을 기다립니다.')

    def open_records(self):
        RUNTIME.mkdir(parents=True, exist_ok=True)
        os.startfile(ROOT / 'artifacts')

    def open_logs(self):
        directory = ROOT / 'artifacts/program-logs'
        directory.mkdir(parents=True, exist_ok=True)
        os.startfile(directory)

    def refresh(self):
        counters = self.counter_reader.snapshot
        if counters:
            self.learning_counts.set(counters['text'])
            stale = self.counter_reader.error or time.monotonic() - counters['at'] > 15
            self.counts_note.set('최근 확인값 · 통계 갱신 재시도 중' if stale else
                f"검증 제외 기록 {counters['evidence_warnings']}건 · 상세 도표 확인" if counters['evidence_warnings'] else
                '약 3초마다 갱신 · 실제 사용된 학습 자료만 집계')
        elif self.counter_reader.error:
            self.counts_note.set('통계 확인 재시도 중 · 학습 실행과 별개')
        try:
            running = worker_running()
            status_path, status = latest_status()
            if running:
                phase = str(status.get('phase', '시작 준비'))
                self.state.set('실행 중 · ' + PHASES.get(phase, phase))
            elif time.monotonic() >= self.pending_until:
                state = str(status.get('status', '아직 실행하지 않음'))
                self.state.set('중지됨 · ' + STATES.get(state, state))
            self.start_button.state(['disabled'] if running or time.monotonic() < self.pending_until else ['!disabled'])
            learning = status.get('background_training') or {}
            learning_state = learning.get('status', 'not_started')
            learning_phase = learning.get('phase', '')
            learning_labels = {'not_started': '아직 시작되지 않음', 'launching': '작업자 시작 중',
                               'running': '진행 중', 'completed': '완료', 'failed': '오류',
                               'cancelled': '중단', 'timed_out': '시간 제한 중단'}
            learning_phases = {'waiting_for_combat': '첫 전투 입력 대기',
                               'training_imports': '라이브러리 준비', 'training_evidence': '자료 검증',
                               'training': '가중치 갱신', 'training_save': '후보 저장',
                               'training_result': '결과 저장', 'training_recovery': '저장된 후보 복원'}
            learning_text = learning_labels.get(learning_state, learning_state)
            if learning_phase:
                learning_text += ' · ' + learning_phases.get(learning_phase, learning_phase)
            online = status.get('online_learning') or {}
            if online:
                online_status = str(online.get('status', '경험 수집'))
                online_labels = {'collecting': '경험 묶음 수집', 'training': '가중치 갱신', 'learning': '가중치 갱신',
                                 'candidate_ready': '새 모델 적용 대기', 'closed': '구간 기록 완료',
                                 'failed': '오류', 'running': '수집·학습 반복'}
                learning_text = ('온라인 · ' + online_labels.get(online_status, online_status)
                                 + f" · 이번 판 모델 반영 {len(online.get('adoptions', []))}회")
            split = {'train': '학습 자료 수집', 'evaluation': '평가 · 학습 자료와 분리'}.get(status.get('split'), '준비')
            if status.get('mode') == 'laya_choice_learning':
                split = 'Laya 전투 전술·성장·상점 학습 · 로컬 실행'
                update = status.get('laya_last_update') or {}
                learning_text = {'updated': '판단 가중치 갱신·저장 완료', 'rejected': '후보 거절 · 이전 모델 유지',
                                 'no_update': '학습 가능한 선택·결과 수집 중'}.get(update.get('status'), '선택·결과 수집 중')
            self.details.set(f"현재 판: {split}\n"
                             f"게임 설정: {(counters or {}).get('focus_settings', '확인 중')}\n"
                             f"별도 학습 기록: {learning_text}\n"
                             f"완료한 비교: {status.get('completed_cycles', status.get('cycle', 0))}\n"
                             f"최근 오류: {status.get('error', status.get('reason', '없음'))}\n"
                             f"상태 기록: {status_path or '없음'}")
        except OSError as error:
            self.details.set('상태 파일 확인 실패: ' + str(error))
        self.window.after(1000, self.refresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('PlayModel.LocalLearning')
    window = tk.Tk()
    def callback_error(kind, value, tb):
        exception('gui_callback_exception', value.with_traceback(tb))
        messagebox.showerror('프로그램 오류', f'{kind.__name__}: {value}\n실행 로그에 기록했습니다.')
    window.report_callback_exception = callback_error
    if args.smoke_test:
        window.withdraw()
    app = App(window)
    if args.smoke_test:
        window.update_idletasks()
        window.destroy()
        print('LOCAL_APP_UI_VERIFIED')
        return
    if args.autostart:
        window.after(500, app.start)
    window.mainloop()


if __name__ == '__main__':
    main()
