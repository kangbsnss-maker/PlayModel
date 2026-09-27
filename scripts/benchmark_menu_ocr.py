"""Read-only cold/persistent OCR comparison on fixed saved game frames."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from playmodel.games.brotato.ocr import MenuOcr, read_menu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve().with_name('windows_ocr.ps1')
    rows = []
    with MenuOcr(script, cache_seconds=0) as worker:
        for image in args.images:
            start = time.perf_counter_ns()
            cold = read_menu(image, script)
            cold_ms = (time.perf_counter_ns() - start) / 1e6
            start = time.perf_counter_ns()
            warm = worker.read(image)
            warm_ms = (time.perf_counter_ns() - start) / 1e6
            rows.append({'image': str(image.resolve()),
                         'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                         'cold_ms': cold_ms, 'persistent_ms': warm_ms,
                         'same_words_and_geometry': cold['lines'] == warm['lines']})
    report = {'scope': 'offline_ocr_latency_not_gameplay_performance', 'samples': rows,
              'all_outputs_equal': all(row['same_words_and_geometry'] for row in rows),
              'cold_median_ms': statistics.median(row['cold_ms'] for row in rows),
              'persistent_median_ms': statistics.median(row['persistent_ms'] for row in rows),
              'first_persistent_includes_startup': True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['all_outputs_equal']:
        raise SystemExit('OCR output parity failed')


if __name__ == '__main__':
    main()
