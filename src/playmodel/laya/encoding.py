"""Preserve complete choice text in Laya's marker-based input layout."""


def encode_complete(tok, instruction, options, state_ids, *, max_len=512):
    """Use upstream token layout without its silent 48-token option truncation."""
    mask = tok.mask_token
    head = tok('choice question: ' + instruction.replace(mask, ' '),
               add_special_tokens=False)['input_ids']
    ids = [tok.cls_token_id, *head, tok.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.append(tok.mask_token_id)
        ids.extend(tok(' ' + option.replace(mask, ' '),
                       add_special_tokens=False)['input_ids'])
    ids.extend([tok.sep_token_id, *state_ids, tok.sep_token_id])
    if len(ids) > max_len:
        raise ValueError(f'Laya complete input needs {len(ids)} tokens; context budget is {max_len}; no truncation allowed')
    return ids, markers
