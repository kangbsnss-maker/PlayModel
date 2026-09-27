"""Evidence-weighted UI transition graph, separate from neural policy learning.

Only arrow transitions are learned here. Enter, purchases, and game actions
remain under the existing screen/target authorization boundary.
"""
import hashlib
import heapq
import json
import time
from pathlib import Path

ARROWS = ('left', 'right', 'up', 'down')


class UiNavigationMemory:
    def __init__(self, ledger, scope, *, learning=True):
        self.ledger, self.scope = Path(ledger), scope
        self.learning = learning
        self.counts, self.durations, self.pending = {}, {}, None
        self.hints = {('difficulty', 'danger_0', 'left'): 'danger_6'}
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding='utf8').splitlines():
                row = json.loads(line)
                if row.get('scope') == scope and row.get('kind') == 'observed_transition':
                    self._count(row)

    def _count(self, row):
        if row.get('key') not in ARROWS:
            raise ValueError('UI transition ledger contains a non-direction input')
        key = (row['scene'], row['from'], row['key'])
        counts = self.counts.setdefault(key, {})
        counts[row['to']] = counts.get(row['to'], 0) + 1
        elapsed = row.get('elapsed_seconds', 0)
        if isinstance(elapsed, (float, int)) and 0 <= elapsed <= 30:
            total, count = self.durations.get(key, (0., 0))
            self.durations[key] = (total + elapsed, count + 1)

    def _write(self, row):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open('a', encoding='utf8') as stream:
            stream.write(json.dumps({'schema': 'playmodel.ui-transition.v1', 'scope': self.scope,
                                     'bc_label': False, **row}, ensure_ascii=False) + '\n')

    @staticmethod
    def _proof(shot):
        source = Path(shot['session_directory']) / 'frame.png'
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != shot['frame_sha256']:
            raise ValueError('UI navigation source hash changed')
        observed, available = shot['capture_started_at_ns'], shot['available_at_ns']
        if (type(observed) is not int or type(available) is not int
                or not 0 < observed <= available <= time.perf_counter_ns()):
            raise ValueError('Noncausal UI navigation observation')
        return {'path': str(source.resolve()), 'sha256': shot['frame_sha256'],
                'observed_at_ns': shot['capture_started_at_ns'], 'available_at_ns': shot['available_at_ns'],
                'identity': {key: shot.get(key) for key in ('hwnd', 'pid', 'executable')}}

    def propose(self, scene, start, target, fallback):
        if start == target or fallback not in ARROWS or not start or not target:
            return fallback
        edges = {}
        for (which, origin, key), counts in self.counts.items():
            if which != scene:
                continue
            dest = max(counts, key=counts.get)
            probability = counts[dest] / sum(counts.values())
            if dest != origin and probability >= .75:
                total, count = self.durations.get((which, origin, key), (0., 0))
                edges.setdefault(origin, []).append((dest, key,
                    (1 + (total / count if count else 0)) / probability))
        for (which, origin, key), dest in self.hints.items():
            # Human-supplied edges are hypotheses. One contradictory observed
            # transition disables the hint; it is never counted as experience.
            if which == scene and (which, origin, key) not in self.counts:
                edges.setdefault(origin, []).append((dest, key, 1.25))
        queue, costs = [(0., start, '')], {}
        while queue:
            cost, node, first = heapq.heappop(queue)
            if node in costs:
                continue
            costs[node] = cost
            if node == target:
                if first and first not in ARROWS:
                    raise ValueError('UI graph cannot authorize confirmation')
                return first or fallback
            for destination, key, weight in edges.get(node, []):
                heapq.heappush(queue, (cost + weight, destination, first or key))
        return fallback

    def sent(self, scene, selected, key, shot, sent_at_ns):
        if key not in ARROWS or selected is None:
            return
        if self.pending is not None:
            raise ValueError('UI navigation has an unobserved prior arrow')
        before = self._proof(shot)
        if not 0 < before['observed_at_ns'] <= before['available_at_ns'] < sent_at_ns:
            raise ValueError('UI arrow observation must precede transmission')
        hint = self.hints.get((scene, selected, key))
        self.pending = {'scene': scene, 'from': selected, 'key': key, 'before': before,
                        'sent_at_ns': sent_at_ns, 'first_after': None,
                        'hint_source': 'user_instruction_unverified' if hint else None,
                        'hint_destination': hint}

    def observe(self, scene, selected, shot):
        """False requests another frame; no key may be sent while awaiting proof."""
        pending = self.pending
        if pending is None:
            return True
        if scene != pending['scene']:
            if self.learning:
                self._write({'kind': 'unresolved_scene_change', **pending, 'next_scene': scene})
            self.pending = None
            return True
        if selected is None:
            return False
        after = self._proof(shot)
        before, first = pending['before'], pending['first_after']
        if after['identity'] != before['identity']:
            raise ValueError('UI navigation target changed')
        if after['observed_at_ns'] <= pending['sent_at_ns']:
            return False
        if first is None or first['selected'] != selected:
            pending['first_after'] = {**after, 'selected': selected}
            return False
        if after['observed_at_ns'] <= first['available_at_ns']:
            return False
        row = {'kind': 'observed_transition', 'scene': scene, 'from': pending['from'],
               'key': pending['key'], 'to': selected, 'before': before,
               'sent_at_ns': pending['sent_at_ns'], 'after': [first, after],
               'elapsed_seconds': (after['available_at_ns'] - pending['sent_at_ns']) / 1e9,
               'hint_source': pending['hint_source'], 'hint_destination': pending['hint_destination'],
               'learning_kind': 'observed_UI_transition_counts_not_neural_policy_gradient'}
        if self.learning:
            self._write(row)
            self._count(row)
        self.pending = None
        return True
