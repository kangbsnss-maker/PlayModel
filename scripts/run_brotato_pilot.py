"""Run with the installed project environment; importing sends no game input.

Example (execute only after confirming an actual combat screen):
  python scripts/run_brotato_pilot.py --exe ".../Brotato.exe" --seconds 30 --stop-file artifacts/STOP

Training additionally needs --train --combat-entry verified-entry.json
--terminal-rules calibrated-terminal-rules.json. Missing/uncertain terminal
evidence preserves an aborted, non-training session. F8 or the stop file stops.
There are no blind menu/retry actions; pause/unknown screens release movement.
On this Windows host launch via the configured hidden-process helper.
"""

if __name__ == "__main__":
    from _execution_bootstrap import launch
    launch(__file__)


from playmodel.games.brotato.pilot import main


if __name__ == "__main__":
    raise SystemExit(main())
