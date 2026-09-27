"""Local bounded session; no GPT/API calls. F8 or artifacts/BROTATO_STOP stops."""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)

from playmodel.games.brotato.session import main

if __name__ == "__main__":
    raise SystemExit(main())
