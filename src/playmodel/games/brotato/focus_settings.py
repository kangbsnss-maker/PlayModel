"""Read-only background-play compatibility; never modify game saves or settings."""
import json
import os
from pathlib import Path


def focus_pause_status(appdata=None):
    base = appdata if appdata is not None else os.environ.get('APPDATA')
    if not base:
        return '게임 포커스 설정 확인 불가'
    try:
        paths = list((Path(base) / 'Brotato').glob('*/settings.json'))
        if len(paths) != 1:
            return '게임 포커스 설정 확인 불가 · 프로필 선택 필요'
        settings = json.loads(paths[0].read_text(encoding='utf-8'))['settings']
        mode = settings.get('on_lost_focus')
        if type(mode) is int and mode in (0, 1, 2):
            enabled = mode != 0
        elif 'on_lost_focus' not in settings and type(settings.get('pause_on_focus_lost')) is bool:
            enabled = settings['pause_on_focus_lost']
        else:
            return '게임 포커스 설정 확인 불가 · 미지원 설정값'
        if enabled:
            return '자동 일시정지 켜짐 · 게임 메인 메뉴 → Options → Sound → On losing focus → Do nothing'
        return '포커스 상실 자동 일시정지 꺼짐'
    except (OSError, ValueError, KeyError, TypeError):
        return '게임 포커스 설정 확인 불가 · 다시 확인 중'
