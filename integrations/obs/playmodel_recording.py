"""Load inside OBS Scripts later. Disabled by default; no network or GPT calls."""
import json
import time
import obspython as obs

enabled = False
state_path = ""
owned = False
session_id = None
blocked = False
started_at = 0.0
pause_attempts = 0


def script_description():
    return "PlayModel: record only combat and level-up scenes. Disabled by default."


def script_defaults(settings):
    obs.obs_data_set_default_bool(settings, "enabled", False)


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_bool(props, "enabled", "Enable recording control")
    obs.obs_properties_add_path(props, "state_path", "PlayModel recording-state.json",
                                obs.OBS_PATH_FILE, "JSON (*.json)", None)
    return props


def script_update(settings):
    global enabled, state_path
    enabled = obs.obs_data_get_bool(settings, "enabled")
    state_path = obs.obs_data_get_string(settings, "state_path")


def script_load(settings):
    script_update(settings)
    obs.timer_add(tick, 100)


def script_unload():
    global owned
    obs.timer_remove(tick)
    if owned and obs.obs_frontend_recording_active():
        obs.obs_frontend_recording_stop()
    owned = False


def tick():
    global owned, session_id, blocked, started_at, pause_attempts
    now = time.time()
    state = {}
    if enabled and state_path:
        try:
            with open(state_path, encoding="utf-8") as stream:
                state = json.loads(stream.read(65536))
        except (OSError, ValueError, TypeError):
            state = {}
    valid = (isinstance(state, dict) and state.get("schema") == "playmodel.obs-scene.v1"
             and state.get("enabled") is True and type(state.get("updated_at")) in (int, float)
             and 0 <= now - state["updated_at"] <= 3)
    current = state.get("session_id") if valid else session_id
    active = obs.obs_frontend_recording_active()
    if current != session_id and not owned:
        session_id, blocked = current, False
    if owned and (not enabled or (valid and state.get("session_open") is False)):
        if active:
            obs.obs_frontend_recording_stop()
        owned = False
        return
    allowed = (valid and state.get("session_open") is True
               and state.get("scene") in ("combat", "level_up"))
    if not owned:
        if active:
            blocked = True  # Never take over an existing/manual OBS recording.
        if allowed and not active and not blocked:
            obs.obs_frontend_recording_start()
            owned, started_at = True, now
        return
    if not active:
        if now - started_at > 2:
            owned, blocked = False, True  # User stop or failed start: no restart loop.
        return
    paused = obs.obs_frontend_recording_paused()
    if allowed:
        pause_attempts = 0
        if paused:
            obs.obs_frontend_recording_pause(False)
    elif not paused:
        obs.obs_frontend_recording_pause(True)
        pause_attempts += 1
        if pause_attempts >= 3:
            # A format/encoder that cannot pause must not record excluded scenes.
            obs.obs_frontend_recording_stop()
            owned, blocked = False, True
    else:
        pause_attempts = 0
