"""Evidence-labelled filenames; unknown metadata stays unknown."""
import json
from pathlib import Path
import re


def part(value):
    text = re.sub(r'[^\w .+-]+','-',str(value),flags=re.UNICODE).strip(' .-')
    return text[:40] or 'Unknown'


def label_recording(directory: Path, context: dict) -> dict:
    manifest = directory/'recording.json'
    report = json.loads(manifest.read_text(encoding='utf-8'))
    source = Path(report['output_path']).resolve()
    allowed = (directory/'original').resolve()
    if source.parent != allowed or not source.is_file():
        raise ValueError('Recording source must be this session original')
    fields = ['Brotato',context.get('character','Unknown-Character'),context.get('difficulty','Unknown-Difficulty'),
              'Endless' if context.get('endless_verified') else 'Mode-Unknown',
              context.get('concept','Unknown-Concept'),'+'.join(context.get('weapons',[])) or 'Unknown-Weapons',
              'Wave-'+str(context.get('max_wave_observed') or 'Unknown'),directory.name]
    title = '__'.join(part(value) for value in fields)
    target = allowed/(title+source.suffix)
    if target != source:
        if target.exists(): raise FileExistsError(target)
        source.rename(target)
    report.update(output_path=str(target),video_title=title,run_context=context)
    manifest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
