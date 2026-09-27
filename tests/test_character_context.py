import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from playmodel.games.brotato.character_context import affinity, modifiers, validated_context, validate_choice_character
from playmodel.games.brotato.setup_run import locked_character, CHARACTER_GRID, focused_tile


class CharacterContextTests(unittest.TestCase):
    def test_signed_traits_condition_weapons_and_passives(self):
        traits=['+20%RangedDamage', '-100%MeleeDamage', '+5MaxHP']
        self.assertGreater(affinity(traits,'Ranged Damage'), affinity(traits,'Melee Damage'))
        self.assertGreater(affinity(traits,'+3 Max HP'), affinity(traits,'-3 Max HP'))
        self.assertEqual(affinity(['Can equip 12 weapons'], 'Melee Damage'),0)
        self.assertEqual(modifiers(['+4Engineering'])['engineering'],'+4engineering')
        self.assertNotIn('range',modifiers(['+20%RangedDamage']))
        self.assertEqual(affinity(['+10Range'],'RangedDamage'),0)
        self.assertEqual(affinity(['+10Range'],'-5%RangedDamage +3Range'),1)
        self.assertEqual(affinity(['+5Speed'],'+10%AttackSpeed'),0)

    def test_source_mutation_rejected_and_missing_context_unknown(self):
        self.assertEqual(validated_context({}),{})
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'frame'
            p.write_bytes(b'frame')
            c={'character_source':str(p),'character_source_sha256':hashlib.sha256(b'frame').hexdigest(),
               'character':'Ranger','traits':['+20%RangedDamage']}
            self.assertEqual(validated_context(c)['name'],'Ranger')
            proof=validated_context({**c,'character_observed_at_ns':10})
            self.assertEqual(validate_choice_character(proof,20)['sha256'],c['character_source_sha256'])
            with self.assertRaises(ValueError): validate_choice_character(proof,5)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): validated_context(c)
            with self.assertRaises(ValueError): validate_choice_character(proof,20)

    def test_locked_focus_needs_condition_and_layout_not_dimness_alone(self):
        pixels=bytearray(1920*1080*4)
        l,t,r,b=CHARACTER_GRID[27]
        for y in range(t,b):
            pixels[(y*1920+l)*4:(y*1920+r)*4]=bytes([122,124,126,255])*(r-l)
        self.assertIsNone(focused_tile(pixels,1920,CHARACTER_GRID))
        def rows(_,rect):
            return {(500,60,1450,160):['CharacterSelection'],
                    (440,445,750,550):['Recycle12weapons','duringarun'],
                    (760,380,1120,530):['Records','NOT_SET']}.get(rect,[])
        with patch('playmodel.games.brotato.setup_run.rows_in_region',side_effect=rows):
            found=locked_character(pixels,1920,{})
        self.assertEqual(found['slot'],27)
        self.assertEqual(found['objective']['count'],12)
        with patch('playmodel.games.brotato.setup_run.rows_in_region',return_value=[]):
            self.assertIsNone(locked_character(pixels,1920,{}))
