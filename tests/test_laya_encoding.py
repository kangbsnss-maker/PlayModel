import unittest

from playmodel.laya.encoding import encode_complete


class Tokenizer:
    mask_token = '[MASK]'
    cls_token_id, sep_token_id, mask_token_id = 1, 2, 3

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(char) + 10 for char in text]}


class LayaEncodingTests(unittest.TestCase):
    def test_long_option_keeps_last_critical_effect_and_markers(self):
        tok = Tokenizer()
        first = 'x' * 60 + ' -5 armor'
        ids, markers = encode_complete(tok, 'choose', [first, 'save'], [900], max_len=512)
        self.assertEqual(ids[markers[0] + 1:markers[1]], tok(' ' + first)['input_ids'])
        self.assertEqual(ids[markers[1]], tok.mask_token_id)
        self.assertEqual(ids[-3:], [tok.sep_token_id, 900, tok.sep_token_id])

    def test_budget_overflow_raises_instead_of_erasing_state(self):
        with self.assertRaisesRegex(ValueError, 'no truncation allowed'):
            encode_complete(Tokenizer(), 'choose', ['x' * 60], [900], max_len=20)

    def test_mask_text_cannot_inject_extra_choice_marker(self):
        ids, markers = encode_complete(Tokenizer(), 'choose', ['use [MASK] item'], [])
        self.assertEqual(len(markers), 1)
        self.assertEqual(ids.count(Tokenizer.mask_token_id), 1)


if __name__ == '__main__':
    unittest.main()
