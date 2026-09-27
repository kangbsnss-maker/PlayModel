"""Label finalized local videos from run evidence, without touching game input."""
import html
import json
import os
from pathlib import Path

from .atomic_io import atomic_json
from .media_titles import part

VERSION = 1


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label_completed(root: Path) -> list[dict]:
    """Keep original paths valid. Descriptive names are same-file hard links."""
    root = root.resolve()
    indexed = {}
    for report in (root / 'artifacts/recurrent-cycles').glob('*/*/segments/*/report.json'):
        operation = report.parents[2] / 'run-operation.json'
        if not operation.is_file():
            continue
        run = _read(operation)
        run_id = report.parents[2].name
        if run.get('partial'):
            purpose = '복구-학습평가제외'
        elif run.get('split') == 'train':
            purpose = '학습판'
        elif run_id.startswith('evaluation-') and '-candidate-' in run_id:
            purpose = '후보모델평가'
        elif run_id.startswith('evaluation-') and '-source-' in run_id:
            purpose = '기준모델평가'
        else:
            purpose = '평가-역할미확인'
        indexed[report.parent.name] = (report, run_id, purpose)
    for report in (root / 'artifacts/run-setup').glob('*/report.json'):
        indexed.setdefault(report.parent.name, (report, None, '시작조건설정-학습아님'))

    changed = []
    for token, (evidence, run_id, purpose) in indexed.items():
        directory = root / 'media/captures' / token
        marker = directory / 'learning-title.json'
        if marker.exists() or not (directory / 'recording.json').is_file():
            continue
        recording = _read(directory / 'recording.json')
        if recording.get('closed') is not True or recording.get('stop_error'):
            continue
        original = Path(recording.get('output_path', '')).resolve()
        if original.parent != (directory / 'original').resolve() or not original.is_file():
            continue
        # This collector currently trains only sparse survival/growth choices.
        # Do not label missing HP, healing or projectile objectives as learned.
        context = recording.get('run_context', {})
        stage = 'S00-기초생존성장' if run_id else '준비'
        title = '__'.join(part(value) for value in (
            stage, purpose, 'Brotato', context.get('character', '캐릭터미확인'),
            context.get('difficulty', '난이도미확인'),
            'Wave-' + str(context.get('max_wave_observed') or 'Unknown'), token))
        alias = original.with_name(title + original.suffix)
        if alias != original:
            if alias.exists():
                if not os.path.samefile(original, alias):
                    raise FileExistsError(alias)
            else:
                os.link(original, alias)
        item = dict(version=VERSION, title=title, stage=stage, purpose=purpose,
                    run_id=run_id, evidence_path=str(evidence),
                    original_path=str(original), display_path=str(alias),
                    same_file_link=True, original_preserved=True)
        # The original output_path stays valid for immutable session reports.
        recording.update(video_title=title, display_path=str(alias),
                         learning_stage=stage, recording_purpose=purpose)
        atomic_json(directory / 'recording.json', recording)
        atomic_json(marker, item)
        changed.append(item)
    if changed:
        rows = []
        for marker in sorted((root / 'media/captures').glob('*/learning-title.json'), reverse=True):
            item = _read(marker)
            link = Path(item['display_path']).as_uri()
            rows.append(f'<li><a href="{html.escape(link, quote=True)}">{html.escape(item["title"])}</a></li>')
        page = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                '<title>PlayModel 학습 영상</title><style>body{max-width:1100px;margin:40px auto;'
                'font:16px/1.7 system-ui;padding:20px}li{margin:14px 0;overflow-wrap:anywhere}</style>'
                '<h1>학습·평가 영상</h1><p>학습판·평가·복구·시작 설정을 구분합니다. '
                '제목은 수행 단계이며 해당 능력의 습득 완료를 뜻하지 않습니다. 원본은 보존됩니다.</p><ul>'
                + ''.join(rows) + '</ul></html>')
        (root / 'media/learning-videos.html').write_text(page, encoding='utf-8')
    return changed
