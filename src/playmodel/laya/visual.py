"""Evidence-bound image windows for the local decision learner."""
import hashlib
import json
from pathlib import Path
import uuid

from .records import digest


def snapshot(evidence, output):
    import torch
    from torch.nn import functional as F
    path = Path(evidence['frame_ref'])
    if digest(path) != evidence['frame_sha256']:
        raise ValueError('visual source frame hash mismatch')
    if path.suffix == '.bgra':
        observation = Path(evidence['observation_path'])
        if digest(observation) != evidence['observation_sha256']:
            raise ValueError('visual source metadata hash mismatch')
        metadata = json.loads(observation.read_text(encoding='utf8'))['metadata']
        width, height = metadata['sample_width'], metadata['sample_height']
        pixels = path.read_bytes()
    else:
        # Capture format decoding belongs to this adapter, not the neural net.
        from playmodel.games.brotato.capture import read_diagnostic_png
        pixels, width, height = read_diagnostic_png(path)
    if width < 1 or height < 1 or len(pixels) != width * height * 4:
        raise ValueError('invalid visual source geometry')
    bgra = torch.frombuffer(bytearray(pixels), dtype=torch.uint8).reshape(height, width, 4)
    crops = []
    if path.suffix == '.bgra':
        document = json.loads(observation.read_text(encoding='utf8'))
        world = document['situation'].get('world', {})
        regions = [(row['position'], max(.025, min(.10, row['radius'] * 2)))
                   for row in world.get('tracks', [])[:3]]
        if world.get('player'):
            regions.append((world['player'], .10))
        for position, radius in regions[:4]:
            cx, cy = int(position[0]*width), int(position[1]*height)
            extent = max(2, int(radius*height))
            box = [max(0,cx-extent), max(0,cy-extent), min(width,cx+extent), min(height,cy+extent)]
            x0,y0,x1,y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            crop = bgra[y0:y1,x0:x1,[2,1,0]].permute(2,0,1).unsqueeze(0)
            crop = F.interpolate(crop.float(), size=(32,32), mode='area').round().to(torch.uint8)
            raw_crop = crop.contiguous().numpy().tobytes()
            crop_path = Path(output) / ('object-' + uuid.uuid4().hex + '.rgb')
            crop_path.write_bytes(raw_crop)
            crops.append({'path': str(crop_path.resolve()), 'sha256': hashlib.sha256(raw_crop).hexdigest(),
                          'bbox': box, 'source_sha256': evidence['frame_sha256'],
                          'transform': 'source_crop_rgb_area32_v1'})
    rgb = bgra[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0)
    rgb = F.interpolate(rgb.float(), size=(96, 96), mode='area').round().to(torch.uint8)
    raw = rgb.contiguous().numpy().tobytes()
    target = Path(output) / ('visual-' + uuid.uuid4().hex + '.rgb')
    target.write_bytes(raw)
    return {'path': str(target.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
            'source_path': str(path.resolve()), 'source_sha256': evidence['frame_sha256'],
            'observed_at_ns': evidence['observed_at_ns'], 'available_at_ns': evidence['available_at_ns'],
            'transform': 'bgra_to_rgb_area96_round_uint8_v1', 'object_crops': crops}


def validate_window(window, evidence):
    if not isinstance(window, list) or not 1 <= len(window) <= 4:
        raise ValueError('one to four visual frames required')
    previous = 0
    for row in window:
        observed, available = row['observed_at_ns'], row['available_at_ns']
        if (type(observed) is not int or type(available) is not int
                or not previous < observed <= available <= evidence['available_at_ns']
                or not 0 <= evidence['observed_at_ns'] - observed <= 2_500_000_000
                or row['transform'] != 'bgra_to_rgb_area96_round_uint8_v1'
                or digest(row['path']) != row['sha256']
                or digest(row['source_path']) != row['source_sha256']):
            raise ValueError('visual window changed, noncausal, or expired')
        previous = observed
        for crop in row.get('object_crops', []):
            if (crop.get('source_sha256') != row['source_sha256']
                    or crop.get('transform') != 'source_crop_rgb_area32_v1'
                    or digest(crop['path']) != crop['sha256']):
                raise ValueError('Object crop evidence changed')
    last = window[-1]
    if (last['source_sha256'] != evidence['frame_sha256']
            or last['observed_at_ns'] != evidence['observed_at_ns']
            or last['available_at_ns'] != evidence['available_at_ns']):
        raise ValueError('visual window does not end at the current observation')


def tensor_window(window, device):
    import torch
    frames = []
    for row in window:
        raw = Path(row['path']).read_bytes()
        if len(raw) != 3 * 96 * 96 or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('visual tensor changed or malformed')
        frames.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(3, 96, 96))
    return torch.stack(frames).unsqueeze(0).to(device)


def object_tensor(window, device):
    import torch
    crops = window[-1].get('object_crops', [])
    if not crops:
        return None
    tensors = []
    for row in crops:
        raw = Path(row['path']).read_bytes()
        if len(raw) != 3*32*32 or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('Object tensor changed or malformed')
        tensors.append(torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(3,32,32))
    return torch.stack(tensors).unsqueeze(0).to(device)
