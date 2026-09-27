"""Start persistent local play coordination and its hidden background learner."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
(root/'artifacts').mkdir(exist_ok=True)
with (root/'artifacts/campaign-launcher.log').open('ab') as log:
    subprocess.Popen([str(root/'.venv/Scripts/pythonw.exe'),'-X','utf8','-m','playmodel.campaign','--record'],
                     cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
