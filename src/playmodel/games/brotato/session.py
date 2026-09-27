"""Bounded local training session with independently checked menu navigation.

Menu choices are explicit bootstrap rules, not learned build optimization.
Shop rerolls are bounded exploration, with verified currency deltas in a separate ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import threading
import time
import uuid
from functools import wraps

from playmodel.learning import StateEvidence, load_checkpoint
from .background import BackgroundController
from .capture import capture_session, read_diagnostic_png
from .installation import inspect_installation
from .menu import classify_scene, selected_button, navigation_key, BUTTONS
from .ocr import MenuOcr, rows_in_region
from .pilot import PilotConfig, TerminalRules, run_pilot
from .vision import BrotatoVision
from .style import ALIASES, load_style, style_digest
from playmodel.video import SessionRecording, make_highlights
from playmodel.instance import session_lock
from playmodel.learning.worker import enqueue, ready_candidate
from playmodel.media_titles import label_recording
from .fast_menu import fast_navigation
from .menu_capture import MenuCapture
from .menu_transition import MenuTransitionGate


def one_session(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with session_lock(Path(__file__).resolve().parents[4] / 'artifacts/brotato-input.lock'):
            return function(*args, **kwargs)
    return wrapped


def upgrade_choice(ocr: dict, style: dict | None = None) -> int:
    """Initial ranged-build rule; no claim of an optimal or learned choice."""
    style = style or load_style()
    scores = []
    for index, left in enumerate((30, 400, 770, 1140)):
        text = "".join(rows_in_region(ocr, (left, 480, left + 360, 600))).casefold()
        score = -1
        if "+" in text and "-" not in text and "−" not in text:
            matches = [(len(label), style['stat_weights'][stat]) for stat, labels in ALIASES.items()
                       for label in labels if label in text]
            if matches:
                score = max(matches)[1]  # Longest label prevents melee damage becoming generic damage.
        scores.append(score)
    if max(scores) < 0:
        raise ValueError("No independently readable ranged-build upgrade")
    return max(range(4), key=lambda index: scores[index])


def calibrated_rules() -> TerminalRules:
    return TerminalRules({
        "schema": "brotato-pilot-terminal-rules-v1", "verified_against_game_build": True,
        "game_build_id": "23429717",
        "calibration_ref": "local Brotato 1.1.15.4 zh/en 1920x1080: 20260926T163710Z-ea89b534, 20260926T164826Z-d99c753a, 20260926T165020Z-6aea8781, 20260926T171633Z-1944c235, 20260926T171904Z-1ca5956e; LeveIup OCR alias: 20260927T060828Z-9e81769c",
        "rules": [
            {'kind':'death','all_text':['RUNLOST','Killedby'],
             'regions':{'RUNLOST':[.3,.12,.7,.26],'Killedby':[.39,.30,.61,.44]}},
            {'kind':'death','all_text':['RUNLOST','KiIIedby'],
             'regions':{'RUNLOST':[.3,.12,.7,.26],'KiIIedby':[.39,.30,.61,.44]}},
            {"kind": "wave_clear", "all_text": ["Shop", "GO"],
             "regions": {"Shop": [0, 0, .25, .12], "GO": [.76, .75, .99, .99]},
             "forbidden_text": ["Resume", "Options"]},
            {"kind": "wave_clear", "all_text": ["Levelup", "Choose"],
             "regions": {"Levelup": [.3, .20, .5, .33], "Choose": [0, .56, .8, .66]},
             "forbidden_text": ["Resume", "Options"]},
            {"kind": "wave_clear", "all_text": ["LeveIup", "Choose"],
             "regions": {"LeveIup": [.3, .20, .5, .33], "Choose": [0, .56, .8, .66]},
             "forbidden_text": ["Resume", "Options"]},
            {"kind": "wave_clear", "all_text": ["ItemFound", "Recycle"],
             "regions": {"ItemFound": [.20, .13, .56, .29], "Recycle": [.20, .68, .56, .78]},
             "forbidden_text": ["Resume", "Options"]},
            {"kind": "wave_clear", "all_text": ["ltemFound", "RecycIe"],
             "regions": {"ltemFound": [.20, .13, .56, .29], "RecycIe": [.20, .68, .56, .78]},
             "forbidden_text": ["Resume", "Options"]},
            {"kind": "wave_clear", "all_text": ["商店", "出发"],
             "regions": {"商店": [0, 0, .25, .12], "出发": [.76, .75, .99, .87]}},
            {"kind": "wave_clear", "all_text": ["发现道具", "回收"],
             "regions": {"发现道具": [.25, .15, .5, .3], "回收": [.2, .68, .55, .78]}},
            {"kind": "wave_clear", "all_text": ["升级", "选择", "刷新"],
             "regions": {"升级": [.3, .20, .5, .33], "选择": [0, .56, .8, .66],
                         "刷新": [.25, .68, .53, .78]}},
        ],
    })


def shop_observation(ocr: dict) -> dict | None:
    """Strict, calibrated numeric regions. Currency icons are outside numeric ROIs."""
    def number(region):
        text = ''.join(rows_in_region(ocr, region))
        return int(text) if re.fullmatch(r'[0-9]{1,5}', text) else None
    currency = number((805, 25, 1020, 110))
    cost = number((1345, 35, 1398, 95))
    header = ''.join(rows_in_region(ocr, (0, 0, 400, 120))).casefold()
    wave = re.search(r'wave([0-9]{1,2})', header)
    if currency is None or wave is None:
        return None
    return {'currency': currency, 'reroll_cost': cost, 'wave': int(wave[1]),
            'offers': [rows_in_region(ocr, (x, 130, x+350, 625)) for x in (25, 386, 747, 1108)],
            'build': rows_in_region(ocr, (1485, 140, 1900, 805))}


def cost_glyph(pixels: bytes, width: int = 1920) -> frozenset:
    """Highlight-invariant glyph geometry, only the calibrated one-digit cost ROI."""
    left, top, right, bottom = 1348, 42, 1385, 88
    corners = [pixels[(y*width+x)*4] for x,y in
               ((left,top),(right-1,top),(left,bottom-1),(right-1,bottom-1))]
    background = sum(corners)/4
    return frozenset((x,y) for y in range(top,bottom) for x in range(left,right)
                     if abs(sum(pixels[(y*width+x)*4:(y*width+x)*4+3])/3-background)>90)


@one_session
def run_session(executable: Path, output: Path, *, waves: int = 2, seconds: float = 180,
                stop_file: Path, ocr_script: Path, checkpoint: Path | None = None,
                style_path: Path | None = None, record: bool = False, edit: bool = True,
                learning_queue: Path | None = None, run_context: dict | None = None,
                combat_runner=None, neural_menu=None) -> dict:
    if not 1 <= waves <= 10 or not 10 <= seconds <= 600:
        raise ValueError("Bounded session requires 1..10 waves and 10..600 seconds")
    if combat_runner is not None and (checkpoint is not None or learning_queue is not None):
        raise ValueError('Experimental combat runner cannot use the linear checkpoint or learning queue')
    if neural_menu is not None and combat_runner is None:
        raise ValueError('Neural menu decisions require a shared neural combat runner')
    directory = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    deadline = time.perf_counter() + seconds
    menu_log, pilots = [], []
    controller = None
    monitor_stop = threading.Event()
    stop_latched = threading.Event()
    style = load_style(style_path)
    latest_policy = output / 'latest-training-policy.json'
    if combat_runner is None and checkpoint is None and latest_policy.exists():
        latest = json.loads(latest_policy.read_text(encoding='utf-8'))
        if latest['style_sha256'] == style_digest(style):
            checkpoint = Path(latest['checkpoint'])
    policy = load_checkpoint(checkpoint) if checkpoint else None
    recording = None
    recording_report = None
    media_directory = Path('media/captures') / directory.name
    shop_log = []
    pending_shop = None
    previous_shop = None
    previous_cost_glyph = None
    navigation_visits = {}
    transition_gate = MenuTransitionGate()
    from .menu_focus import LevelUpFocusRecovery
    focus_recovery = LevelUpFocusRecovery()
    from .neural_navigation import FrozenNavigation
    frozen_navigation = FrozenNavigation()
    learning_jobs, candidate_adoptions = [], []
    context = dict(run_context or {})
    observed_waves = []
    clear_observations = 0
    shop_counts, shop_spend = {}, {}
    shop_reroll_abandoned = set()
    release_error = None
    reason = "menu_limit"
    unknown_attempts = 0
    last_vision = None
    menu_reader = MenuOcr(ocr_script, cache_seconds=0 if neural_menu is not None else 2.0)
    menu_capture = MenuCapture(executable)
    recognition_metrics = []
    fast_model = None
    fast_model_error = None
    model_path = Path(__file__).resolve().parents[4] / 'models/menu/approved.json'
    approval_path = model_path.with_name('approval.json')
    model_build = None
    if model_path.exists() and approval_path.exists():
        try:
            from .menu_model import MenuClassifier
            model_build = next(i['steam_build_id'] for i in inspect_installation()['installations']
                               if Path(i['path']).resolve() == executable.parent.resolve())
            fast_model = MenuClassifier.load(model_path, approval_path=approval_path)
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
            fast_model_error = str(error)

    def check():
        if stop_file.exists() or stop_latched.is_set():
            raise OSError("Session stopped")
        if time.perf_counter() >= deadline:
            raise OSError("Session time limit")
        if recording is not None and recording.error:
            raise OSError('Recording failed: ' + recording.error)
        if controller is not None:
            controller.check()

    def observe():
        started = time.perf_counter_ns()
        check()
        shot, pixels, width, height = menu_capture.read(directory / 'menus', check=check)
        image = Path(shot["session_directory"]) / "frame.png"
        if (width, height) != (1920, 1080):
            raise ValueError("This menu calibration requires 1920x1080")
        fast = None
        if neural_menu is not None and not neural_menu.awaiting_application:
            fast = frozen_navigation.propose(neural_menu.pending_decision, shot, pixels,
                                             time.perf_counter_ns())
            if fast is not None:
                recognition_metrics.append({'source': str(image),
                    'path': 'fresh_pixels_frozen_macro_navigation',
                    'observation_ms': (time.perf_counter_ns()-started)/1e6})
                (image.parent / 'recognition.json').write_text(json.dumps(asdict(fast)), encoding='utf-8')
                return shot, image, None, pixels, width, height, fast
        if fast_model is not None and neural_menu is None:
            fast = fast_navigation(fast_model, pixels, width, height, game_build_id=model_build)
            if time.perf_counter_ns() - shot['capture_started_at_ns'] > 500_000_000:
                fast = None
        if fast is not None:
            recognition_metrics.append({'source': str(image), 'path': 'approved_local_menu_model',
                'observation_ms': (time.perf_counter_ns()-started)/1e6})
            (image.parent / 'recognition.json').write_text(json.dumps(asdict(fast)), encoding='utf-8')
            return shot, image, None, pixels, width, height, fast
        ocr = menu_reader.read(image)
        check()
        if (neural_menu is not None and classify_scene(ocr, width=width, height=height).scene == 'shop'
                and not re.fullmatch(r'(?:0|[1-9][0-9]{0,4})', ''.join(rows_in_region(ocr, (805, 25, 1020, 110))))):
            from .shop_currency_ocr import read_shop_currency
            currency = read_shop_currency(menu_reader, image, pixels=pixels,
                observed_at_ns=shot['capture_started_at_ns'], frame_id=str(image.resolve()),
                game_build_id=neural_menu.recorder.game_build_id)
            ocr['_local_shop_currency'] = asdict(currency)
            ocr['primary_available_at_ns'] = ocr['available_at_ns']
            ocr['available_at_ns'] = max(ocr['available_at_ns'], currency.available_at_ns)
            ocr['recognition_path'] = 'persistent_local_ocr_with_currency_roi'
            check()
        if (neural_menu is not None and neural_menu.recorder.model.config.context_dim == 64
                and neural_menu.needs_build_observation):
            stats_scene = classify_scene(ocr, width=width, height=height).scene
            if stats_scene in ('shop', 'level_up'):
                from .stats_roi_ocr import read_stats_roi
                stats = read_stats_roi(menu_reader, image, pixels=pixels,
                    observed_at_ns=shot['capture_started_at_ns'], frame_id=str(image.resolve()),
                    game_build_id=neural_menu.recorder.game_build_id, scene=stats_scene)
                ocr['_local_stats'] = stats
                ocr.setdefault('primary_available_at_ns', ocr['available_at_ns'])
                ocr['available_at_ns'] = max(ocr['available_at_ns'], stats['available_at_ns'])
                ocr['recognition_path'] = 'persistent_local_ocr_with_state_roi'
                check()
        (image.parent / "ocr.json").write_text(json.dumps(ocr, ensure_ascii=False, indent=2), encoding="utf-8")
        recognition_metrics.append({'source': str(image), 'path': ocr['recognition_path'],
            'ocr_ms': (ocr['available_at_ns']-ocr['processing_started_at_ns'])/1e6,
            'observation_ms': (time.perf_counter_ns()-started)/1e6})
        return shot, image, ocr, pixels, width, height, None

    def monitor():
        while not monitor_stop.wait(.02):
            if controller is not None and controller._context.key_state(0x77) & 0x8000:
                stop_latched.set()
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()

    watcher = threading.Thread(target=monitor, daemon=True, name="brotato-session-stop")
    watcher.start()
    try:
        if record:
            recording = SessionRecording(media_directory, max_seconds=seconds + 30)
            recording.event('session_start', '로컬 모델 학습을 시작합니다. 게임 화면과 메뉴 전 과정을 기록합니다.')
        (directory / 'style.json').write_text(json.dumps(style, ensure_ascii=False, indent=2), encoding='utf-8')
        for _ in range(300 if neural_menu is not None else 80):
            shot, source, ocr, pixels, width, height, fast = observe()
            if controller is None:
                controller = BackgroundController(shot["hwnd"], executable)
            elif controller.hwnd != shot["hwnd"]:
                raise OSError("Game window changed")
            if not transition_gate.allow(shot['frame_sha256'],
                                         captured_at_ns=shot['capture_started_at_ns']):
                continue
            if fast is not None:
                decision_id = getattr(fast, 'decision_id', None)
                signature = (fast.scene, fast.selected, fast.target, len(pilots), len(shop_log), decision_id)
                navigation_visits[signature] = navigation_visits.get(signature, 0) + 1
                if navigation_visits[signature] > 3:
                    raise OSError('Repeated menu navigation without progress')
                check()
                if time.perf_counter_ns()-shot['capture_started_at_ns'] > 500_000_000:
                    continue
                controller.tap_menu(fast.key)
                sent_at = time.perf_counter_ns()
                frozen_navigation.sent(sent_at)
                transition_gate.record(shot['frame_sha256'], fast.key, posted_at_ns=sent_at)
                from playmodel.execution_log import event
                event('menu_input_sent', scene=fast.scene, target=fast.target, selected=fast.selected,
                      key=fast.key, frame_path=str(source), sent_at_ns=sent_at,
                      recognition_path='fresh_pixels_frozen_macro_navigation' if decision_id else 'approved_local_menu_model',
                      decision_id=decision_id, game_application_verified=False)
                menu_log.append({'source': str(source), 'sha256': shot['frame_sha256'],
                    'scene': fast.scene, 'selected': fast.selected, 'target': fast.target,
                    'key': fast.key, 'posted_at_ns': time.perf_counter_ns(),
                    'perception_origin': 'fresh_pixels_frozen_macro_navigation' if decision_id else 'approved_local_menu_model',
                    'model_label': fast.model_label, 'decision_id': decision_id,
                    'policy_origin': 'verified_navigation_for_neural_choice' if decision_id else 'bootstrap_rule',
                    'learned_choice': False})
                continue
            scene = classify_scene(ocr, width=width, height=height).scene
            neural_directive = None
            # A partial recovery is never training/evaluation score data. It
            # may finish an unsupported loot screen using the existing rule;
            # a normal learned run still fails closed instead of inventing a
            # policy decision or its likelihood for that unrecorded choice.
            recovery_loot = scene == 'loot' and context.get('partial_recovery') is True
            if neural_menu is not None and scene != 'unknown':
                neural_directive = neural_menu.handle(shot, ocr, pixels, width=width, height=height)
                recognition_metrics[-1]['neural_menu_directive'] = asdict(neural_directive)
                if neural_directive.status in ('wait', 'accepted'):
                    continue
                if (neural_directive.status == 'unhandled' and scene in ('shop', 'level_up', 'loot')
                        and not recovery_loot):
                    raise OSError('Neural menu scene unsupported; no rule-based substitution')
            header = ''.join(rows_in_region(ocr,(0,0,1920,90)))
            wave_match = re.search(r'wave\s*(\d{1,3})',header,re.I)
            if wave_match:
                observed_waves.append(int(wave_match[1]))
            if scene=='shop':
                shop_header=''.join(rows_in_region(ocr,(0,0,400,120)))
                next_header=''.join(rows_in_region(ocr,(1450,780,1910,1070)))
                past=re.search(r'wave\s*(\d+)',shop_header,re.I)
                upcoming=re.search(r'wave\s*(\d+)',next_header,re.I)
                if past and upcoming and int(past[1])>=20 and int(upcoming[1])==int(past[1])+1:
                    clear_observations+=1
                    if clear_observations==1 and not context.get('stage_clear_criterion_met'):
                        controller.release()
                        continue
                    if clear_observations>=2:
                        context.update(stage_clear_criterion_met=True,stage_clear_source=str(source),
                                       stage_clear_rule='post_wave_20_shop_and_next_wave_confirmed',
                                       continue_endless_after_clear=True)
                else:
                    clear_observations=0
                if learning_queue is not None and deadline-time.perf_counter()<140:
                    reason='segment_limit'
                    break
            if pending_shop is not None:
                after = shop_observation(ocr) if scene == 'shop' else None
                before = pending_shop['before']
                confirmed = (after is not None and after['wave'] == before['wave']
                             and before['currency'] - after['currency'] == before['reroll_cost']
                             and before['offers'] != after['offers'])
                pending_shop.update(after=after, after_source=str(source), applied_verified=confirmed,
                                    causal_outcome='pending_episode_outcome')
                shop_log.append(pending_shop)
                pending_shop = None
                if not confirmed:
                    raise OSError('Shop reroll outcome unverified; no repeated spending')
                if recording:
                    recording.event('shop_reroll', f"상점을 갱신했습니다. 재료 {before['reroll_cost']}개를 사용했고 상품 변화가 확인됐습니다.")
            if scene == "unknown":
                # Vision already samples a bounded <=240x135 grid. Reducing
                # this fresh capture first changes the sampling lattice and
                # can split the white body into equally sized fragments.
                vision = BrotatoVision().observe(pixels, width, height,
                                                  observed_at_ns=shot["capture_started_at_ns"])
                last_vision = asdict(vision)
                if not vision.combat_likely or vision.player is None:
                    unknown_attempts += 1
                    from playmodel.execution_log import event
                    event('game_scene_unrecognized', frame_path=str(source), scene=scene,
                          vision=last_vision, observation_attempt=unknown_attempts)
                    if unknown_attempts <= 3:
                        # Reobserve without input through short spawn/transition animations.
                        controller.release()
                        continue
                    raise OSError("Unrecognized screen; no action")
                unknown_attempts = 0
                evidence = StateEvidence("combat", str(source.resolve()), shot["frame_sha256"],
                                         shot["capture_started_at_ns"], shot["available_at_ns"],
                                         time.perf_counter_ns(), "local_detector",
                                         "brotato-hud-player-v1", True, True)
                if neural_menu is not None:
                    neural_menu.combat_entry(evidence)
                remaining = deadline - time.perf_counter()
                if remaining <= 5:
                    raise OSError("Insufficient remaining session time")
                if learning_queue is not None and policy is not None:
                    ready = ready_candidate(learning_queue, policy.version, style_digest(style))
                    if ready is not None:
                        policy = load_checkpoint(ready['checkpoint'])
                        candidate_adoptions.append(ready['job_id'])
                        (output/'latest-training-policy.json').write_text(json.dumps({
                            'checkpoint':ready['checkpoint'],'style_sha256':style_digest(style),
                            'promotion_approved':False,'experimental_training_continuation':True}),encoding='utf-8')
                        if recording:
                            recording.event('training_update','백그라운드 후보 모델을 다음 전투 실험에 적용합니다. 수치 검사 통과, 실력 향상 미검증.')
                if recording:
                    recording.event('combat_start', '전투 시작. 로컬 이동 정책이 화면을 보고 이동합니다. 아이템 수집과 위험 회피를 시도합니다.')
                menu_capture.close()
                # Neural sessions already warmed this uncached reader during
                # menu recognition. Its lock serializes any final terminal read
                # with the next menu read; the observer only borrows ownership.
                combat_options = {'terminal_ocr_reader': menu_reader} if neural_menu is not None else {}
                pilot = (combat_runner or run_pilot)(executable, directory / "pilots", policy=policy,
                                  config=PilotConfig(max_seconds=min(130, remaining), max_steps=2600, train=combat_runner is None,
                                                     defer_training=learning_queue is not None, movement_hold_ms=200),
                                  stop_file=stop_file, combat_entry=evidence,
                                  terminal_rules=calibrated_rules(), ocr_script=ocr_script,
                                  vision_factory=lambda: BrotatoVision(avoidance_radius=style['behavior']['avoidance_radius']),
                                  **combat_options)
                pilots.append(pilot)
                if learning_queue is not None and pilot.get('training_deferred'):
                    learning_jobs.append(enqueue(learning_queue, Path(pilot['session_directory']), style_sha=style_digest(style)))
                    if policy is None:
                        policy = load_checkpoint(Path(pilot['session_directory'])/'initial-policy.json')
                    if recording:
                        recording.event('learning_queued','전투 기록을 백그라운드 학습 대기열에 추가했습니다.')
                for event in shop_log:
                    if event['causal_outcome'] == 'pending_episode_outcome':
                        event['causal_outcome'] = 'episode_link_only_not_causal_credit'
                        event['next_pilot'] = pilot['session_directory']
                        event['next_result'] = pilot['reason']
                if recording:
                    recording.event('wave_result', '전투 구간 종료. 결과를 별도 화면 인식으로 검증합니다.', reason=pilot['reason'])
                if pilot["status"] == "aborted":
                    reason = pilot["reason"]
                    break
                if pilot["training_performed"]:
                    policy = load_checkpoint(Path(pilot["session_directory"]) / "candidate-policy.json")
                    candidate = str((Path(pilot['session_directory']) / 'candidate-policy.json').resolve())
                    (output / 'latest-training-policy.json').write_text(json.dumps({'checkpoint': candidate,
                        'style_sha256': style_digest(style), 'promotion_approved': False}), encoding='utf-8')
                    if recording:
                        recording.event('training_update', '이번 전투 기록으로 이동 모델 가중치를 갱신했습니다. 실력 향상 여부는 아직 평가하지 않았습니다.')
                if len(pilots) >= waves:
                    reason = "wave_limit"
                    break
                continue
            if scene == "pause":
                target = "continue"
            elif scene == 'death':
                target = 'ok'
                if recording:
                    recording.event('run_lost','사망이 확인됐습니다. 이번 판의 기록을 저장합니다.')
            elif scene == 'result':
                reason = 'run_finished'
                context['result_source'] = str(source)
                context['result_text'] = header
                break
            elif scene == 'difficulty':
                target = 'danger_6'
                if 'Nightmare' in ''.join(rows_in_region(ocr,(1450,185,1740,245))):
                    context['difficulty'] = 'Nightmare'
                    context['difficulty_source'] = str(source)
            elif neural_directive is not None and neural_directive.status == 'navigate':
                target = neural_directive.target
            elif scene == "shop":
                target = 'depart'
                observed = shop_observation(ocr)
                if (observed is not None and observed['reroll_cost'] is None and previous_shop is not None
                        and previous_shop['reroll_cost'] is not None and previous_shop['reroll_cost'] < 10
                        and previous_cost_glyph and cost_glyph(pixels) == previous_cost_glyph
                        and observed['currency'] == previous_shop['currency']
                        and observed['wave'] == previous_shop['wave']):
                    observed['reroll_cost'] = previous_shop['reroll_cost']
                if observed is not None:
                    wave = observed['wave']
                    budget = style['shop']
                    cost = observed['reroll_cost']
                    if previous_shop is not None and previous_shop['wave'] == wave and cost is None:
                        shop_reroll_abandoned.add(wave)
                    can_reroll = (wave not in shop_reroll_abandoned and cost is not None and shop_counts.get(wave, 0) < budget['max_rerolls_per_wave']
                        and observed['currency'] - cost >= budget['reserve']
                        and shop_spend.get(wave, 0) + cost <= (observed['currency'] + shop_spend.get(wave, 0)) * budget['max_budget_fraction'])
                    if can_reroll:
                        if previous_shop != observed:
                            previous_shop = observed
                            previous_cost_glyph = cost_glyph(pixels)
                            continue  # Two agreeing independent observations before spending.
                        target = 'refresh'
                elif previous_shop is not None:
                    # Do not oscillate between reroll and departure when the
                    # highlighted price becomes unreadable. Skip this optional
                    # spending attempt for this wave; never invent its price.
                    shop_reroll_abandoned.add(previous_shop['wave'])
            elif scene == "loot":
                target = "recycle"
            elif scene == "level_up":
                target = "choose_" + str(upgrade_choice(ocr, style))
            else:
                raise OSError("Menu requires additional calibration")
            unknown_attempts = 0
            candidates=dict(BUTTONS[scene])
            if scene=='shop' and any('go' in row.casefold() for row in rows_in_region(ocr,(1450,950,1910,1070))):
                candidates['depart']=(1484,981,1884,1048)
            selection = selected_button(pixels, width, height, scene=scene,candidates=candidates)
            if scene in ('shop', 'loot') and selection.reason == 'no_selected_candidate':
                from .menu import selected_stat_row
                selection=selected_stat_row(pixels,width,height,scene=scene)
            if scene == 'difficulty':
                from .setup_run import focused_tile
                from .menu import ButtonSelection
                boxes={i:BUTTONS['difficulty'][f'danger_{i}'] for i in range(7)}
                index=focused_tile(pixels,width,boxes)
                if index is not None:
                    selection=ButtonSelection(f'danger_{index}',boxes[index],'calibrated_tile_border')
            focus_acquisition = False
            focus_key = None
            if selection.selected_id is None and neural_directive is not None:
                focus_key = focus_recovery.observe(selection, scene=scene,
                    decision_id=neural_directive.decision_id, frame_id=str(source),
                    observed_at_ns=shot['capture_started_at_ns'], now_ns=time.perf_counter_ns())
                if focus_key == 'wait':
                    continue
                focus_acquisition = focus_key == 'left'
            if selection.selected_id is None and not focus_acquisition:
                raise OSError("Menu selection is ambiguous; no input")
            target_rect = candidates[target]
            key = focus_key if focus_acquisition else (
                "enter" if selection.selected_id == target else navigation_key(selection.rect, target_rect, scene=scene))
            if scene == 'difficulty' and key == 'enter':
                label = ''.join(rows_in_region(ocr, (1450,185,1740,245))).casefold()
                if selection.selected_id != 'danger_6' or 'nightmare' not in label:
                    raise OSError('Highest difficulty 6 not verified; no start input')
            if key is None:
                raise OSError("No verified navigation direction")
            signature = (scene, selection.selected_id, target, len(pilots), len(shop_log),
                         neural_directive.decision_id if neural_directive is not None else None)
            check()
            if focus_acquisition and time.perf_counter_ns() - shot['capture_started_at_ns'] > 750_000_000:
                continue
            if time.perf_counter_ns() - shot["capture_started_at_ns"] > 2_500_000_000:
                raise OSError("Menu observation expired")
            neural_action = neural_directive is not None and neural_directive.status == 'navigate'
            if neural_action and key != 'enter' and not focus_acquisition:
                frozen_navigation.arm(neural_menu.pending_decision, shot, pixels, candidates,
                                      time.perf_counter_ns())
            if neural_action and key == 'enter':
                if not neural_menu.authorize_enter(neural_directive.decision_id, selection.selected_id,
                                                   shot, ocr, pixels):
                    from playmodel.execution_log import event
                    event('menu_input_deferred', scene=scene, target=target, frame_path=str(source),
                          decision_id=neural_directive.decision_id,
                          reason='enter_not_authorized_reobserve', transmitted=False)
                    continue
            # Count actual input attempts, not observations that were deferred
            # because their evidence expired before authorization.
            navigation_visits[signature] = navigation_visits.get(signature, 0) + 1
            if navigation_visits[signature] > 3:
                raise OSError('Repeated menu navigation without progress')
            send_started_at_ns = time.perf_counter_ns()
            controller.tap_menu(key)
            sent_at_ns = time.perf_counter_ns()
            frozen_navigation.sent(sent_at_ns)
            if focus_acquisition:
                focus_recovery.mark_sent(neural_directive.decision_id)
            transition_gate.record(shot['frame_sha256'], key, posted_at_ns=sent_at_ns)
            if neural_action and key == 'enter':
                neural_menu.mark_sent(sent_at_ns=sent_at_ns, actual_target=target,
                                      send_started_at_ns=send_started_at_ns)
            from playmodel.execution_log import event
            event('menu_input_sent', scene=scene, target=target, selected=selection.selected_id,
                  key=key, frame_path=str(source), send_started_at_ns=send_started_at_ns,
                  sent_at_ns=sent_at_ns, focus_acquisition=focus_acquisition,
                  game_application_verified=False)
            if scene == 'shop' and target == 'refresh' and key == 'enter' and not neural_action:
                wave, cost = observed['wave'], observed['reroll_cost']
                shop_counts[wave] = shop_counts.get(wave, 0) + 1
                shop_spend[wave] = shop_spend.get(wave, 0) + cost
                pending_shop = {'before': observed, 'source': str(source), 'sha256': shot['frame_sha256'],
                                'posted_at_ns': time.perf_counter_ns(), 'style_sha256': style_digest(style),
                                'policy_origin': 'bounded_exploration_not_learned'}
                previous_shop = None
            if recording and key == 'enter':
                message = {'level_up': f"성장 카드 {int(target[-1]) + 1}번 선택 입력을 보냈습니다. 설정한 능력치 가중치 기준입니다." if scene == 'level_up' else '',
                           'death':'사망 결과 확인 입력을 보냈습니다.',
                           'difficulty':'최고 난이도 Nightmare로 새 판을 시작합니다.',
                           'loot': '아이템 재활용 입력을 보냈습니다. 현재는 초기 규칙을 사용합니다.',
                           'shop': ('상품 갱신 입력을 보냈습니다. 재료 차감과 상품 변화를 확인합니다.' if target == 'refresh'
                                    else '상점 검토를 마치고 다음 웨이브 출발 입력을 보냈습니다.'), 'pause': '플레이 재개 입력을 보냈습니다.'}[scene]
                if neural_action:
                    message = '로컬 신경망이 선택한 행동을 전송했습니다. 실제 적용 결과를 확인합니다.'
                recording.event('menu_choice', message, scene=scene, target=target, applied_verified=False)
            menu_log.append({"source": str(source), "sha256": shot["frame_sha256"], "scene": scene,
                             "selected": selection.selected_id, "target": target, "key": key,
                             "posted_at_ns": time.perf_counter_ns(),
                             "policy_origin": "verified_navigation_recovery" if focus_acquisition else (
                                 "recurrent_neural_policy" if neural_action else "bootstrap_rule"),
                             "learned_choice": neural_action and not focus_acquisition,
                             "focus_acquisition": focus_acquisition,
                             **({'recovery_only': True, 'training_eligible': False,
                                 'bootstrap_reason': 'unsupported_loot_in_excluded_partial_recovery'}
                                if recovery_loot and not neural_action else {})})
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        from playmodel.execution_log import event, exception
        exception('game_session_failed', error)
        event('game_session_failure_context', error=reason,
              session_directory=str(directory.resolve()),
              frame_path=str(source) if 'source' in locals() else None,
              scene=scene if 'scene' in locals() else None, vision=last_vision,
              vision_from_current_frame=bool(last_vision and 'shot' in locals()
                  and last_vision.get('observed_at_ns') == shot.get('capture_started_at_ns')))
    finally:
        menu_capture.close()
        menu_reader.close()
        monitor_stop.set()
        watcher.join(.1)
        if controller is not None:
            try:
                controller.release()
            except Exception as error:
                release_error = str(error)
            try:
                final = capture_session(executable, directory / "menus", timeout=4)
                small, sw, sh = read_diagnostic_png(Path(final["session_directory"]) / "frame.png", stride=6)
                state = BrotatoVision().observe(small, sw, sh)
                if state.combat_likely and not stop_latched.is_set() and not stop_file.exists():
                    controller.tap_menu("escape")
            except Exception:
                pass
        if recording is not None:
            recording.event('stop', '학습 세션을 종료합니다. 원본을 보존하고 주요 장면과 녹음용 자막을 만듭니다.', reason=reason)
            time.sleep(1)
            try:
                recording_report = recording.close()
            except Exception as error:
                recording_report = {'error': str(error), 'review_needed': True}
                reason = 'recording_shutdown_unconfirmed'
    report = {"session_directory": str(directory.resolve()), "reason": reason, "pilots": pilots,
              "training_updates": sum(bool(p["training_performed"]) for p in pilots),
              "menu_choices": "bootstrap_rules_not_learned", "automatic_shop_purchases": False,
              "build_calibration": "Brotato 1.1.15.4 en/zh 1920x1080", "stop_file": str(stop_file),
              'style': style, 'recording': recording_report, 'release_error': release_error,
              'shop_rerolls': len(shop_log), 'checkpoint_input': str(checkpoint) if checkpoint else None}
    report.update(background_learning_jobs=learning_jobs, experimental_candidate_adoptions=candidate_adoptions)
    report['combat_policy_backend'] = 'experimental_external_runner' if combat_runner is not None else 'linear_movement'
    if neural_menu is not None:
        report['menu_choices'] = 'recurrent_neural_upgrade_loot_and_shop_experimental'
        report['automatic_shop_purchases'] = True
    report['menu_recognition_metrics'] = recognition_metrics
    report['menu_model_load_error'] = fast_model_error
    report['shop_reroll_abandoned_waves'] = sorted(shop_reroll_abandoned)
    report['run_context'] = {**context, 'concept':style['name'],
                             'max_wave_observed':max(observed_waves+[context.get('max_wave_observed') or 0]) or None}
    if recording_report and recording_report.get('output_path'):
        try:
            report['recording'] = recording_report = label_recording(media_directory,report['run_context'])
        except Exception as error:
            report['video_title_error'] = str(error)
    (directory / 'shop-learning.json').write_text(json.dumps(shop_log, ensure_ascii=False, indent=2), encoding='utf-8')
    if reason not in ('wave_limit','segment_limit','run_finished'):
        (directory / 'help-request.json').write_text(json.dumps({'reason': reason, 'automatic_agent_call': False,
            'last_menu_source': str(source) if 'source' in locals() else None, 'recoveries_exhausted': unknown_attempts,
            'review_needed': True}, ensure_ascii=False, indent=2), encoding='utf-8')
    (directory / "menu-actions.json").write_text(json.dumps(menu_log, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if edit and recording_report and recording_report.get('output_path'):
        try:
            report['edit'] = make_highlights(media_directory)
        except Exception as error:
            report['edit_error'] = str(error)
        (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded local Brotato learning session; F8 stops")
    parser.add_argument("--waves", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("artifacts/brotato-sessions"))
    parser.add_argument("--stop-file", type=Path, default=Path("artifacts/BROTATO_STOP"))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--style', type=Path)
    parser.add_argument('--record', action='store_true', help='Record the entire session through local OBS')
    parser.add_argument('--no-edit', action='store_true')
    parser.add_argument('--learning-queue', type=Path)
    args = parser.parse_args(argv)
    installation = next(i for i in inspect_installation()["installations"] if i["status"] == "files_present")
    report = run_session(Path(installation["path"]) / "Brotato.exe", args.output,
                         waves=args.waves, seconds=args.seconds, stop_file=args.stop_file,
                         ocr_script=Path("scripts/windows_ocr.ps1"), checkpoint=args.checkpoint,
                         style_path=args.style, record=args.record, edit=not args.no_edit,
                         learning_queue=args.learning_queue)
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["reason"] == "wave_limit" else 2
