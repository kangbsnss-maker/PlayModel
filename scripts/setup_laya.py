"""Download a pinned English Laya checkpoint; never called during gameplay."""
from pathlib import Path
import hashlib
import json

REPO = 'convaiinnovations/laya'
REVISION = '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'
FILES = ('model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
         'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json', 'README.md')


def main():
    from huggingface_hub import hf_hub_download
    root = Path(__file__).resolve().parents[1]
    target = root / 'models/laya/base'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=target))
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        hashes[name] = digest
        print(name, path.stat().st_size, digest, flush=True)
    manifest = {'repository': REPO, 'revision': REVISION, 'license': 'Apache-2.0',
                'runtime_package': 'laya==0.3.20', 'files': hashes}
    (target / 'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('LAYA_DOWNLOAD_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
