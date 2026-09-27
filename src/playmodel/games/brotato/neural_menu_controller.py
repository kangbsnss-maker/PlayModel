"""Stateful local neural menu adapter; caller remains the only input writer.

One macro is sampled and kept through navigation. Fresh focus/candidate checks
authorize Enter; its successful transport is logged separately. Two independent
post-action observations (or verified combat entry for skip) precede recorder
append and hidden-state commit. There is no bootstrap-choice fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import torch

from playmodel.learning import StateEvidence
from playmodel.learning.full_run import FullRunRecorder
from .menu import BUTTONS, classify_scene, selected_button
from .ocr import rows_in_region
from .neural_choices import (
    MAX_AGE_NS, ChoiceObservation, FrozenMacroDecision, MacroApplication,
    observe_menu, sample_choice, save_macro_record, stable_loot,
    verify_shop_choice, verify_shop_skip, verify_upgrade_choice, verify_loot_choice,
)
from .shop_learning import ShopObservation, VerifiedShop, observe_shop, stable_shop


class NeuralMenuError(RuntimeError):
    """Stop this action path; never silently substitute a bootstrap choice."""


@dataclass(frozen=True)
class MenuDirective:
    status: str
    target: str | None
    decision_id: str | None
    reason: str


@dataclass
class _Pending:
    decision: FrozenMacroDecision
    before_shop: VerifiedShop | None
    sent_at_ns: int | None = None
    actual_target: str | None = None
    first_after: ChoiceObservation | None = None
    first_shop_after: ShopObservation | None = None
    after_shop: VerifiedShop | None = None
    authorized_frame: str | None = None
    authorized_observed_at_ns: int | None = None
    authorized_at_ns: int | None = None
    navigation_observations: int = 0
    result_observations: int = 0
    last_result_reason: str | None = None
    last_application_check: MacroApplication | None = None


class NeuralMenuController:
    def __init__(self, recorder: FullRunRecorder, *, output_directory: Path | None = None,
                 seed: int = 0, clock: Callable[[], int] = time.perf_counter_ns,
                 max_navigation_observations: int = 48, max_result_observations: int = 10):
        if not isinstance(recorder, FullRunRecorder):
            raise ValueError("Shared FullRunRecorder required")
        if (type(seed) is not int or seed < 0 or type(max_navigation_observations) is not int
                or max_navigation_observations < 2 or type(max_result_observations) is not int
                or max_result_observations < 2):
            raise ValueError("Invalid bounded neural menu settings")
        self.recorder = recorder
        self.output_directory = Path(output_directory).resolve() if output_directory else None
        self.clock = clock
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.max_navigation_observations = max_navigation_observations
        self.max_result_observations = max_result_observations
        self._pending: _Pending | None = None
        self._previous_shop: ShopObservation | None = None
        self._previous_stats = None
        self._previous_loot: ChoiceObservation | None = None
        self._ready_shop: VerifiedShop | None = None
        self._preparation_observations = 0
        self._stale_observations = 0
        self._authorization_expirations = 0
        self._prepared_observation: ChoiceObservation | None = None
        self._prepared_fingerprint = None
        self._prepared_shop = None
        self.failed_reason: str | None = None
        self.samples = 0
        self.accepted = 0
        self.last_application: MacroApplication | None = None
        self._transmission_reports = 0
        self._shop_diagnostics = None

    @property
    def pending_decision(self) -> FrozenMacroDecision | None:
        return self._pending.decision if self._pending else None

    @property
    def awaiting_application(self) -> bool:
        return self._pending is not None and self._pending.sent_at_ns is not None

    @property
    def needs_build_observation(self) -> bool:
        """Stats belong to a new decision; pending macros already froze them.

        This never permits reuse of current button, price or offer OCR. Those
        observations still run on every navigation/authorization frame.
        """
        return self._pending is None

    def _open(self):
        if self.failed_reason:
            raise NeuralMenuError(self.failed_reason)
        try:
            self.recorder._open()
        except (ValueError, RuntimeError) as error:
            self._fail(str(error))

    def _pending_details(self) -> dict | None:
        pending = self._pending
        if pending is None:
            return None
        return {"decision_id": pending.decision.decision_id,
                "target": pending.decision.target,
                "authorized_frame": pending.authorized_frame,
                "authorized_observed_at_ns": pending.authorized_observed_at_ns,
                "authorized_at_ns": pending.authorized_at_ns,
                "sent_at_ns": pending.sent_at_ns, "actual_target": pending.actual_target,
                "result_observations": pending.result_observations,
                "last_result_reason": pending.last_result_reason,
                "last_application_check": (asdict(pending.last_application_check)
                                           if pending.last_application_check is not None else None)}

    def _fail(self, reason: str, *, details: dict | None = None):
        self.failed_reason = str(reason)
        self.recorder.invalidate("neural_menu:" + self.failed_reason)
        if self.output_directory is not None:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            path = self.output_directory / f"failure-{self.clock()}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"reason": self.failed_reason, "pending_decision":
                           self.pending_decision.decision_id if self.pending_decision else None,
                           "pending_state": self._pending_details(), "details": details}, stream)
        raise NeuralMenuError(self.failed_reason)

    def abort(self, reason: str):
        self._fail(reason)

    def _source(self, shot: dict, ocr: dict, pixels: bytes, width: int, height: int) -> dict:
        """Preserve capture time separately from OCR-information availability."""
        try:
            frame = (Path(shot["session_directory"]) / "frame.png").resolve()
            now = self.clock()
            observed = shot["capture_started_at_ns"]
            available = max(shot["available_at_ns"], ocr["available_at_ns"])
            if (type(observed) is not int or type(available) is not int
                    or not 0 < observed <= available <= now or now - observed > MAX_AGE_NS
                    or (width, height) != (1920, 1080) or len(pixels) != width * height * 4
                    or ocr.get("recognition_path") == "exact_image_cache" or "cache_source" in ocr):
                raise ValueError("Fresh uncached menu evidence required")
            if hashlib.sha256(frame.read_bytes()).hexdigest() != shot["frame_sha256"]:
                raise ValueError("Menu source frame modified")
            return dict(pixels=pixels, frame_id=str(frame), observed_at_ns=observed,
                        available_at_ns=available, game_build_id=self.recorder.game_build_id,
                        width=width, height=height)
        except (OSError, KeyError, TypeError, ValueError) as error:
            self._fail(str(error))

    def _shop(self, ocr: dict, source: dict) -> VerifiedShop | None:
        current = observe_shop(ocr, **source, currency_observation=ocr.get('_local_shop_currency'))
        verified = stable_shop(self._previous_shop, current, now_ns=self.clock())
        self._shop_diagnostics = {
            'current': asdict(current) if current is not None else None,
            'previous': asdict(self._previous_shop) if self._previous_shop is not None else None,
            'independent_agreement': verified is not None}
        self._previous_shop = current
        self._ready_shop = verified
        return verified

    def _previous_movement(self) -> int:
        for record in reversed(self.recorder.records):
            if int(record["tensors"]["phase"].item()) == 0:
                return int(record["tensors"]["actions"].item())
        return 0

    def _wait(self, reason: str) -> MenuDirective:
        if self.awaiting_application:
            self._pending.last_result_reason = reason
        return MenuDirective("wait", None, self.pending_decision.decision_id if self.pending_decision else None, reason)

    def _defer_stale_observation(self, *, stage='handle', source=None, shot=None) -> MenuDirective:
        """Reobserve without authorizing input or discarding the sampled macro."""
        self._stale_observations += 1
        if stage.startswith('authorize'):
            # A fresh handle() must not erase repeated expiry immediately
            # before Enter. Count this separately until a real authorization.
            self._authorization_expirations += 1
        observed = source.get('observed_at_ns') if source else (shot or {}).get('capture_started_at_ns')
        frame = source.get('frame_id') if source else str(Path(shot['session_directory']) / 'frame.png') if shot else None
        pending = self._pending
        from playmodel.execution_log import event
        event('neural_menu_deferred', reason='stale_menu_observation_reobserve_without_input',
              stage=stage, frame_path=frame,
              observation_age_ns=self.clock() - observed if type(observed) is int else None,
              max_observation_age_ns=MAX_AGE_NS, stale_observations=self._stale_observations,
              authorization_expirations=self._authorization_expirations,
              decision_id=pending.decision.decision_id if pending else None,
              proposed_target=pending.decision.target if pending else None,
              actual_target=pending.actual_target if pending else None,
              transmitted=pending is not None and pending.sent_at_ns is not None)
        if self._authorization_expirations > self.max_result_observations:
            self._fail('Fresh Enter authorization remained unavailable',
                       details={'stage': stage, 'authorization_expirations': self._authorization_expirations,
                                'frame_path': frame, 'max_observation_age_ns': MAX_AGE_NS})
        if self._stale_observations > self.max_result_observations:
            self._fail("Fresh menu observations remained unavailable")
        self._previous_shop = self._ready_shop = None
        self._previous_loot = None
        self._prepared_observation = self._prepared_fingerprint = self._prepared_shop = None
        if self._pending is not None:
            pending = self._pending
            pending.authorized_frame = pending.authorized_observed_at_ns = pending.authorized_at_ns = None
            pending.first_after = pending.first_shop_after = None
        return self._wait("stale_menu_observation_reobserve_without_input")

    @staticmethod
    def _observation_fingerprint(shot, ocr, source):
        """Bind one prepared immutable observation to ALL authorization inputs.

        Source PNG is rehashed by _source on every call. Pixels and whole OCR
        (including ROI provenance/availability) must also remain identical.
        A matching title or button alone is never a reusable observation.
        """
        ocr_digest = hashlib.sha256(json.dumps(ocr, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        return (source['frame_id'], shot['frame_sha256'], source['observed_at_ns'],
                source['available_at_ns'], source['game_build_id'], source['width'], source['height'],
                hashlib.sha256(source['pixels']).hexdigest(), ocr_digest)

    def _candidate_same(self, current: ChoiceObservation) -> bool:
        pending = self._pending
        before = pending.decision.observation
        if current.scene != before.scene or len(current.candidates) != len(before.candidates):
            return False
        if any(old.semantic_id != new.semantic_id or old.target != new.target
               for old, new in zip(before.candidates, current.candidates)):
            return False
        chosen = current.candidates[pending.decision.action_index]
        if not chosen.legal or chosen.target != pending.decision.target:
            return False
        if before.scene == "shop":
            if (current.wave, current.currency, current.weapon_fill) != (before.wave, before.currency, before.weapon_fill):
                return False
            # Price is an observed constraint. Same semantic item at a new price
            # does not authorize execution of the old sampled purchase.
            if before.candidates[pending.decision.action_index].features[12:14] != chosen.features[12:14]:
                return False
        return True

    def handle(self, shot: dict, ocr: dict, pixels: bytes, width: int = 1920, height: int = 1080) -> MenuDirective:
        """Called for each fresh observation. ``wait`` always means no input."""
        self._open()
        self._prepared_observation = self._prepared_fingerprint = self._prepared_shop = None
        if self._pending is not None and self._pending.sent_at_ns is None:
            self._pending.authorized_frame = None
            self._pending.authorized_observed_at_ns = None
            self._pending.authorized_at_ns = None
        observed = shot.get("capture_started_at_ns")
        if type(observed) is int and observed > 0 and self.clock() - observed > MAX_AGE_NS:
            return self._defer_stale_observation(stage='handle_capture', shot=shot)
        source = self._source(shot, ocr, pixels, width, height)
        scene = classify_scene(ocr, width, height).scene
        if self.awaiting_application:
            return self._after_input(scene, ocr, source)
        if self._pending is not None and scene != self._pending.decision.observation.scene:
            self._fail("Menu changed before the sampled macro was transmitted")
        if scene not in ("level_up", "shop", "loot"):
            self._previous_shop = self._ready_shop = None
            self._previous_loot = None
            return MenuDirective("unhandled", None, None, "not_a_supported_learned_choice")
        verified_shop = self._shop(ocr, source) if scene == "shop" else None
        if self.recorder.build_state is not None and self.needs_build_observation and scene in ('shop', 'level_up'):
            from .state_features import make_stats_observation
            stats = ocr.get('_local_stats')
            if stats is None:
                stats = make_stats_observation(ocr, frame_ref=source['frame_id'],
                    pixels_sha256=hashlib.sha256(pixels).hexdigest(),
                    observed_at_ns=source['observed_at_ns'], available_at_ns=source['available_at_ns'],
                    scene=scene, game_build_id=self.recorder.game_build_id)
            previous_stats = self._previous_stats
            if previous_stats is not None:
                self.recorder.build_state.observe_stats_pair(self._previous_stats, stats)
            if verified_shop is not None and self._pending is None:
                # Prefer the dedicated numeric ROI before the coarser full-frame
                # shop text. Inventory observation still updates in this call.
                self.recorder.build_state.observe_shop_pair(verified_shop)
            self._previous_stats = stats
            if previous_stats is None and self._pending is None and scene == 'level_up':
                if self.clock() - source['observed_at_ns'] > MAX_AGE_NS:
                    return self._defer_stale_observation(stage='handle_stats', source=source)
                return self._wait('verify_build_stats_with_independent_observation')
        if scene == "shop" and verified_shop is None:
            if self.clock() - source['observed_at_ns'] > MAX_AGE_NS:
                return self._defer_stale_observation(stage='handle_shop', source=source)
            self._preparation_observations += 1
            if self._preparation_observations > self.max_navigation_observations:
                self._fail("Shop never produced two independent agreeing observations",
                           details=self._shop_diagnostics)
            return self._wait("verify_shop_with_independent_observation")
        try:
            current = observe_menu(ocr, shop=verified_shop, **source)
        except ValueError as error:
            self._fail(str(error))
        if scene == 'loot' and self._pending is None:
            agreed = stable_loot(self._previous_loot, current)
            self._previous_loot = current
            if not agreed:
                self._preparation_observations += 1
                if self._preparation_observations > self.max_navigation_observations:
                    self._fail('Loot never produced two independent agreeing card/button observations')
                return self._wait('verify_loot_identity_with_independent_observation')
        elif scene != 'loot':
            self._previous_loot = None
        previous_action = self._previous_movement() if self._pending is None else None
        prepared_at = self.clock()
        # ROI/state parsing and image preparation may consume the remaining
        # freshness budget. No sampled action or input uses an expired frame.
        # Only completed fresh preparation resets the consecutive-expiry bound.
        if prepared_at - source['observed_at_ns'] > MAX_AGE_NS:
            return self._defer_stale_observation(stage='handle_preparation', source=source)
        self._stale_observations = 0
        if self._pending is None:
            try:
                decision = sample_choice(self.recorder.model, current, hidden=self.recorder.hidden,
                                         previous_action=previous_action, reset=not self.recorder.records,
                                         generator=self.generator, now_ns=prepared_at,
                                         build_state=self.recorder.build_state)
            except (ValueError, RuntimeError) as error:
                self._fail(str(error))
            self._pending = _Pending(decision, verified_shop)
            self._authorization_expirations = 0
            self.samples += 1
            self._preparation_observations = 0
            if self.output_directory is not None:
                save_macro_record(self.output_directory / decision.decision_id / "proposed", decision)
        else:
            self._pending.navigation_observations += 1
            if self._pending.navigation_observations > self.max_navigation_observations:
                self._fail("Sampled menu target did not become executable")
            if not self._candidate_same(current):
                return self._wait("sampled_candidate_identity_or_legality_unconfirmed")
        pending = self._pending
        pending.authorized_frame = None
        self._prepared_observation = current
        self._prepared_fingerprint = self._observation_fingerprint(shot, ocr, source)
        self._prepared_shop = verified_shop
        return MenuDirective("navigate", pending.decision.target, pending.decision.decision_id,
                             "execute_the_existing_sampled_macro")

    def authorize_enter(self, decision_id: str, selected_target: str, shot: dict, ocr: dict,
                         pixels: bytes, width: int = 1920, height: int = 1080) -> bool:
        """Return True only when Enter may be sent now; False means reobserve.

        The caller MUST skip input on False. Freshness is checked again after
        expensive source/candidate/focus verification, immediately before return.
        """
        self._open()
        pending = self._pending
        if (pending is None or pending.sent_at_ns is not None or pending.decision.decision_id != decision_id
                or selected_target != pending.decision.target):
            self._fail("Enter does not match the pending sampled macro")
        # Reauthorization must never leave an older successful approval active.
        pending.authorized_frame = pending.authorized_observed_at_ns = pending.authorized_at_ns = None
        observed = shot.get("capture_started_at_ns")
        if type(observed) is int and observed > 0 and self.clock() - observed > MAX_AGE_NS:
            self._defer_stale_observation(stage='authorize_capture', shot=shot)
            return False
        source = self._source(shot, ocr, pixels, width, height)
        cached = self._prepared_observation
        fingerprint = self._observation_fingerprint(shot, ocr, source)
        if cached is not None and source['frame_id'] == cached.frame_id and fingerprint != self._prepared_fingerprint:
            self._fail('Prepared menu evidence changed before Enter')
        if (cached is not None and fingerprint == self._prepared_fingerprint
                and (cached.scene != 'shop' or self._ready_shop is self._prepared_shop)):
            # Reuse only work done for THIS exact fresh frame. Enter still
            # rechecks source bytes, candidate identity, pixel focus and age.
            observation, scene = cached, cached.scene
        else:
            scene = classify_scene(ocr, width, height).scene
            shop = None
            if scene == "shop":
                # OCR may have changed even when its frame path has not. Never
                # pair cached shop prices with a different OCR digest.
                shop = self._shop(ocr, source)
                if shop is None:
                    self._fail("Enter requires independently revalidated shop legality")
            try:
                observation = observe_menu(ocr, shop=shop, **source)
            except ValueError as error:
                self._fail(str(error))
        if not self._candidate_same(observation):
            self._fail("Candidate changed or became illegal before Enter")
        boxes = dict(BUTTONS[scene])
        if scene == "shop" and any("go" in text.casefold() for text in rows_in_region(ocr, (1450, 950, 1910, 1070))):
            boxes["depart"] = (1484, 981, 1884, 1048)
        selection = selected_button(pixels, width, height, scene=scene, candidates=boxes)
        if selection.selected_id != selected_target:
            self._fail("Latest pixel focus does not match the sampled Enter target")
        authorized_at = self.clock()
        if authorized_at - source["observed_at_ns"] > MAX_AGE_NS:
            self._defer_stale_observation(stage='authorize_verification', source=source)
            return False
        pending.authorized_frame = source["frame_id"]
        pending.authorized_observed_at_ns = source["observed_at_ns"]
        pending.authorized_at_ns = authorized_at
        self._authorization_expirations = 0
        return True

    def mark_sent(self, *, sent_at_ns: int, actual_target: str,
                  send_started_at_ns: int | None = None) -> None:
        """Record successful transport before validation; application is separate.

        Capture send_started_at_ns immediately before tap_menu and sent_at_ns
        immediately after it returns successfully, in the same perf-counter
        clock domain. Reporting/recorder validation latency must not retroactively
        expire input that started with fresh evidence. If the optional start is
        absent, completion is conservatively used for the freshness check.
        """
        pending = self._pending
        reported_at = self.clock()
        started = sent_at_ns if send_started_at_ns is None else send_started_at_ns
        failures = []
        if pending is None:
            failures.append("no_pending_decision")
        else:
            if pending.sent_at_ns is not None:
                failures.append("duplicate_transmission")
            if actual_target != pending.decision.target:
                failures.append("target_mismatch")
            if pending.authorized_at_ns is None or pending.authorized_observed_at_ns is None:
                failures.append("missing_enter_authorization")
        if type(started) is not int or type(sent_at_ns) is not int:
            failures.append("non_integer_transport_clock")
        elif not 0 < started <= sent_at_ns <= reported_at:
            failures.append("unordered_transport_clock")
        elif pending is not None and pending.authorized_at_ns is not None:
            if started < pending.authorized_at_ns:
                failures.append("transmission_predates_authorization")
            if (pending.authorized_observed_at_ns is not None
                    and started - pending.authorized_observed_at_ns > MAX_AGE_NS):
                failures.append("observation_expired_before_transmission")
        details = {"schema": "playmodel.neural-menu-transport.v1",
                   "send_started_at_ns": started, "sent_at_ns": sent_at_ns,
                   "send_start_measured": send_started_at_ns is not None,
                   "reported_at_ns": reported_at, "actual_target": actual_target,
                   "successful_transport_reported": True, "game_application_verified": False,
                   "guard_failures": failures, "authorization": self._pending_details(),
                   "max_observation_age_ns": MAX_AGE_NS}
        from playmodel.execution_log import event
        event('neural_menu_input_reported', decision_id=pending.decision.decision_id if pending else None,
              proposed_target=pending.decision.target if pending else None,
              actual_target=actual_target, sent_at_ns=sent_at_ns,
              send_started_at_ns=started, successful_transport_reported=True,
              game_application_verified=False, guard_failures=failures)
        if pending is not None and pending.authorized_observed_at_ns is not None:
            details["observation_age_at_report_ns"] = reported_at - pending.authorized_observed_at_ns
            if type(started) is int:
                details["observation_age_at_send_start_ns"] = started - pending.authorized_observed_at_ns
        if self.output_directory is not None:
            directory = self.output_directory / (pending.decision.decision_id if pending else "unmatched-transmissions")
            directory.mkdir(parents=True, exist_ok=True)
            self._transmission_reports += 1
            # Keep every reported successful transport, even a rejected/duplicate
            # report. Never make it disappear behind a later freshness/model check.
            path = directory / f"transmission-{self._transmission_reports:04d}-{reported_at}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(details, stream)
        if failures:
            self._fail("Unapproved, duplicate or stale macro transmission: " + ", ".join(failures), details=details)
        pending.sent_at_ns, pending.actual_target = sent_at_ns, actual_target
        if self.output_directory is not None:
            path = self.output_directory / pending.decision.decision_id / "sent.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"sent_at_ns": sent_at_ns, "actual_target": actual_target,
                           "send_started_at_ns": started, "reported_at_ns": reported_at,
                           "authorized_frame": pending.authorized_frame,
                           "authorization": self._pending_details(), "game_application_verified": False}, stream)
        # A changed/closed recorder still invalidates learning, but the actual
        # transmission receipt above survives and this macro cannot be resent.
        self._open()

    def _after_input(self, scene: str, ocr: dict, source: dict) -> MenuDirective:
        pending = self._pending
        pending.result_observations += 1
        if pending.result_observations > self.max_result_observations:
            self._fail("Menu action application remained unverified; no repeated Enter")
        if source["observed_at_ns"] <= pending.sent_at_ns:
            self._fail("Post-action capture predates the actual Enter transmission")
        before = pending.decision.observation
        if before.scene == "shop":
            chosen = before.candidates[pending.decision.action_index]
            if chosen.candidate_id == "skip":
                return self._wait("await_independent_combat_entry_for_shop_skip")
            if scene != "shop":
                return self._wait("await_shop_action_observation")
            current = observe_shop(ocr, **source, currency_observation=ocr.get('_local_shop_currency'))
            after = stable_shop(pending.first_shop_after, current, now_ns=self.clock())
            pending.first_shop_after = current
            if after is None:
                return self._wait("verify_shop_action_with_independent_observation")
            application = verify_shop_choice(pending.decision, pending.before_shop, after,
                                               sent_at_ns=pending.sent_at_ns, actual_target=pending.actual_target,
                                               now_ns=self.clock())
            pending.after_shop = after
        else:
            if scene == "unknown":
                return self._wait("await_readable_post_choice_menu")
            try:
                current = observe_menu(ocr, **source)
            except ValueError as error:
                self._fail(str(error))
            first = pending.first_after
            pending.first_after = current
            if first is None:
                return self._wait("verify_loot_with_independent_observation" if before.scene == 'loot'
                                  else "verify_upgrade_with_independent_observation")
            verify = verify_loot_choice if before.scene == 'loot' else verify_upgrade_choice
            application = verify(pending.decision, first, current, sent_at_ns=pending.sent_at_ns,
                                 actual_target=pending.actual_target, now_ns=self.clock())
        pending.last_application_check = application
        if not application.accepted:
            return self._wait(application.reason)
        return self._commit(application)

    def _commit(self, application: MacroApplication) -> MenuDirective:
        pending = self._pending
        try:
            self.recorder.append_macro(pending.decision, application)
        except (OSError, ValueError, RuntimeError) as error:
            self._fail("Full-run recorder rejected menu application: " + str(error))
        if self.output_directory is not None:
            save_macro_record(self.output_directory / pending.decision.decision_id / "accepted",
                              pending.decision, application)
        state = self.recorder.build_state
        if state is not None:
            candidate = pending.decision.observation.candidates[pending.decision.action_index]
            if candidate.candidate_id.startswith('buy:') and pending.after_shop is not None:
                state.apply_verified_purchase(pending.before_shop, pending.after_shop,
                    int(candidate.candidate_id.split(':')[1]), application=application)
            elif pending.decision.observation.scene in ('level_up', 'loot'):
                # Selection acceptance is not evidence of its stat effects.
                state.invalidate_stats(pending.decision.observation.scene + '_selected_reobservation_required')
            self._previous_stats = None
        self.accepted += 1
        self.last_application = application
        decision_id = pending.decision.decision_id
        self._pending = None
        self._previous_shop = self._ready_shop = None
        self._previous_loot = None
        self._preparation_observations = 0
        return MenuDirective("accepted", None, decision_id, application.reason)

    def combat_entry(self, evidence: StateEvidence) -> MenuDirective:
        """Run before starting the next combat collector, so it sees committed memory."""
        self._open()
        if self._pending is None:
            return MenuDirective("unhandled", None, None, "no_pending_menu_macro")
        pending = self._pending
        if pending.sent_at_ns is None:
            self._fail("Combat began before a sampled menu macro was transmitted")
        application = verify_shop_skip(pending.decision, evidence, sent_at_ns=pending.sent_at_ns,
                                        actual_target=pending.actual_target, now_ns=self.clock())
        if not application.accepted:
            self._fail(application.reason)
        return self._commit(application)
