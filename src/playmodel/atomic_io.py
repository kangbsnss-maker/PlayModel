"""Atomic JSON snapshots with bounded Windows reader-sharing retries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid


_REPLACE_DELAYS = (.005, .010, .020, .040, .080)
_WINDOWS_SHARING_ERRORS = frozenset((5, 32, 33))


def atomic_json(path: Path | str, value, *, durable: bool = False) -> None:
    """Readers see a complete old or new document, never a partially written one.

    Windows may report access denied when a reader opens the destination without
    FILE_SHARE_DELETE. Retry only the potentially transient Windows codes, then
    propagate persistent permission/sharing errors. No permissions are changed.
    """
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid.uuid4().hex}.tmp')
    original_error = None
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(content)
            if durable:
                stream.flush()
                os.fsync(stream.fileno())
        for attempt in range(len(_REPLACE_DELAYS) + 1):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                if (getattr(error, 'winerror', None) not in _WINDOWS_SHARING_ERRORS
                        or attempt == len(_REPLACE_DELAYS)):
                    raise
                time.sleep(_REPLACE_DELAYS[attempt])
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f'Temporary snapshot cleanup failed: {cleanup_error}')
