"""Observed UI layers, separate from permission to send an input."""
class UiLayers:
    def __init__(self):
        self.base = None

    def observe(self, scene, *, frame_ref, observed_at_ns, pending_action=False):
        overlay = None
        if scene == 'pause':
            overlay = 'pause'
        elif scene == 'unknown':
            overlay = 'unresolved_visual_change'
        else:
            self.base = scene
        return {'schema': 'playmodel.ui-layers.v1', 'base_scene': self.base,
                'base_scene_is_current': overlay is None, 'overlay': overlay,
                'observed_scene': scene, 'frame_ref': str(frame_ref),
                'observed_at_ns': observed_at_ns, 'pending_action': bool(pending_action),
                'input_authorized': False,
                'note': 'layer observation alone never grants input authority'}
