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
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from playmodel.execution_log import event, exception

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/local/learner-launch.json'
RUNTIME = ROOT / 'artifacts/local-learning'
STOP = ROOT / 'artifacts/BROTATO_STOP'
LOCK = RUNTIME / 'worker.lock'
PHASES = {'initializing': '준비', 'obs_start': 'OBS 실행·연결', 'cycle_start': '다음 학습 준비',
          'new_run_setup': '캐릭터·무기·난이도 선택', 'combat': '전투 경험 수집',
          'menu': '성장·상점 선택', 'training': '가중치 학습',
          'training_worker_started': '별도 학습 작업 준비',
          'await_training_worker': '판 종료 · 학습 결과 대기',
          'comparison_complete': '새 판 비교 완료', 'stopped': '중지',
          'recovery': '진행 중인 판 복구', 'candidate_rejected': '후보 검사 탈락',
          'collect_training': '학습용 새 판 수집', 'training_collected': '학습 판 저장 완료',
          'train_candidate': '후보 모델 학습', 'candidate_saved': '후보 모델 저장 완료',
          'collect_evaluation': '독립 평가 판 수집', 'evaluation_collected': '평가 판 저장 완료',
          'bounded_observation_recovery': '화면 재확인', 'active_run_recovery': '진행 중인 판 마무리'}
STATES = {'fault_paused': '오류 기록 확인 필요', 'user_stopped': '사용자 중지',
          'budget_paused': '실행 예산 종료', 'budget_complete': '요청한 실행 완료',
          'candidate_rejected': '후보 검사 탈락', 'starting': '시작 준비'}


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


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


def launch_worker(checkpoint: Path, resume_summary: Path | None, *, resume_latest=True) -> tuple[int, Path]:
    if worker_running():
        raise RuntimeError('학습 프로그램이 이미 실행 중입니다.')
    if not checkpoint.is_file():
        raise ValueError('모델 파일을 선택하세요.')
    if resume_summary and not resume_summary.is_file():
        raise ValueError('이어서 평가할 summary.json 파일이 없습니다.')
    python = ROOT / '.venv/Scripts/python.exe'
    if not python.is_file():
        raise ValueError('.venv Python 환경이 없습니다. 실행 가이드를 확인하세요.')
    command = [str(python), '-X', 'utf8', str(ROOT / 'scripts/run_recurrent_cycle.py'),
               str(checkpoint.resolve()), '--continuous', '--recover-active-run', '--evaluation-runs', '1',
               '--max-run-seconds', '1800', '--device', 'cuda']
    if resume_latest:
        command += ['--resume-latest']
    if resume_summary:
        command += ['--resume-summary', str(resume_summary.resolve())]
    else:
        command += ['--online-updates']
    # The user explicitly requested resume by pressing Start. Never clear this
    # file during background retries or passive application startup.
    STOP.unlink(missing_ok=True)
    log = RUNTIME / ('worker-' + time.strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns()) + '.log')
    with log.open('ab') as output:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    event('child_started', child_pid=child.pid, program='run_recurrent_cycle.py',
          output_log=str(log), checkpoint=str(checkpoint.resolve()))
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({'checkpoint': str(checkpoint.resolve()),
                                 'resume_summary': str(resume_summary.resolve()) if resume_summary else '',
                                 'resume_latest': True},
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
        window.geometry('790x570')
        window.minsize(650, 530)
        frame = ttk.Frame(window, padding=20)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='PlayModel', font=('Segoe UI', 23, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='게임 플레이 · 경험 수집 · PPO 학습 · 새 판 평가\n정상 실행에는 에이전트나 API 토큰이 필요하지 않습니다.').pack(anchor='w', pady=(4, 16))
        config = read_json(CONFIG)
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
        self.state = tk.StringVar(value='실행 상태 확인 중')
        ttk.Label(frame, textvariable=self.state, font=('Segoe UI', 12, 'bold'), wraplength=720).pack(anchor='w')
        self.details = tk.StringVar()
        ttk.Label(frame, textvariable=self.details, wraplength=720, justify='left').pack(anchor='w', pady=10)
        ttk.Label(frame, text='창을 닫아도 학습은 계속됩니다. 중지는 위 버튼 또는 F8.\n오류가 나면 기록을 보존하고 멈춥니다. 코드 자체 수정은 자동 수행하지 않습니다.',
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

    def start(self):
        try:
            summary = Path(self.resume.get().strip()) if self.resume.get().strip() else None
            pid, _ = launch_worker(Path(self.checkpoint.get().strip()), summary,
                                   resume_latest=self.resume_latest.get())
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
            self.details.set(f"현재 판: {split}\n"
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
