"""Frozen text feature cache; trainable decision layers always run afresh.

Mirrors the pinned Laya DecisionModel choice path, excluding its unused act head.
The cache belongs to one immutable text encoder and stores no logits/gradients.
"""
from collections import OrderedDict


class ChoiceForward:
    def __init__(self, model, *, max_bytes=64 * 1024 * 1024):
        self.model = model
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def __call__(self, tensors, *, key, visual_frames=None, object_frames=None):
        import torch
        model = self.model
        if model.training or any(p.requires_grad for p in model.encoder.parameters()):
            raise ValueError('feature cache requires an eval-mode frozen text encoder')
        h = self.cache.get(key)
        if h is None:
            with torch.no_grad():
                h = model.encoder(input_ids=tensors['input_ids'],
                                  attention_mask=tensors['attention_mask']).last_hidden_state.detach()
            self.misses += 1
            size = h.numel() * h.element_size()
            if size <= self.max_bytes:
                while self.cache and self.bytes + size > self.max_bytes:
                    _, old = self.cache.popitem(last=False)
                    self.bytes -= old.numel() * old.element_size()
                self.cache[key] = h
                self.bytes += size
        else:
            self.hits += 1
            self.cache.move_to_end(key)
        h = h + model.type_emb(tensors['qtype'])[:, None, :]
        if model.head is not None:
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=~tensors['attention_mask'].bool())
        indices = tensors['marker_pos'].clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        candidates = torch.gather(h, 1, indices)
        logits = model.scorer(candidates).squeeze(-1).float()
        if visual_frames is not None:
            logits = logits + model.visual_context(visual_frames, candidates, object_frames).float()
        return logits.masked_fill(~tensors['marker_mask'], -1e4)
