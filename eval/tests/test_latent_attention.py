"""File identity, replay timing and spatial projection regression tests."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from eval.common import write_json
from eval.latent_attention import (image_layout, load_record, numbered_filename,
                                   overlay_image, render_outputs, replay_plan, select_layers)
from eval.model import LatentPolicy, decode_stream


class FakeBackend:
    eos_ids = {2}

    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.calls = []

    def next_token(self, state, banned):
        return next(self.tokens)

    def advance(self, state, token=None):
        self.calls.append(token)
        return state


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 5, 8, 20])
def test_replay_matches_decoder_calls_for_closed_and_truncated_segments(budget):
    # First latent_end is suppressed by min_steps. Later one exits normally.
    backend = FakeBackend([7, 8, 19, 8, 12, 7, 20, 8, 2])
    prediction = decode_stream(backend, None, LatentPolicy(7, 8, 1, 5, True), budget)
    prediction['method'] = 'generation'
    plan = replay_plan(prediction, 7, 8, {2})
    assert [x['token'] for x in plan] == backend.calls
    assert sum(x['token'] is None for x in plan) == sum(x['steps'] for x in prediction['latent_segments'])
    assert [x['decode_step'] for x in plan] == list(range(len(backend.calls)))


def test_forced_end_and_wrong_checkpoint_boundary():
    backend = FakeBackend([7, 11, 11, 11, 2])
    prediction = decode_stream(backend, None, LatentPolicy(7, 8, 1, 2, True), 20)
    prediction['method'] = 'generation'
    assert prediction['latent_segments'][0]['forced_end']
    assert [x['token'] for x in replay_plan(prediction, 7, 8, {2})] == [7, None, None, 8]
    with pytest.raises(ValueError, match='Boundary token mismatch'):
        replay_plan(prediction, 17, 18, {2})
    broken = copy.deepcopy(prediction)
    broken['latent_segments'][0]['end_step'] += 1
    with pytest.raises(ValueError, match='end_step'):
        replay_plan(broken, 7, 8, {2})


def test_likelihood_has_no_latent_attention():
    with pytest.raises(ValueError, match='likelihood'):
        replay_plan({'method': 'likelihood'}, 7, 8, {2})


def test_filename_is_not_internal_id_and_score_contains_its_own_prediction(tmp_path):
    sample = {'id': '1376', 'benchmark': 'mmstar'}
    prediction = {'method': 'generation', 'text': 'D'}
    score = {'id': '1376', 'identity': ['00000001', prediction, sample], 'status': 'ok'}
    write_json(tmp_path / 'scores/00000001/00001365.json', score)
    # Deliberately conflicting same-name prediction: must never be used for a score lookup.
    write_json(tmp_path / 'predictions/00001365.json', {'id': 'wrong'})
    path, _, actual, result = load_record(tmp_path, 'scores', '00000001', 1365, 'mmstar')
    assert path.name == '00001365.json' and actual['id'] == '1376' and result == prediction
    assert numbered_filename('00001365.json') == numbered_filename(1365)
    with pytest.raises(FileNotFoundError):
        load_record(tmp_path, 'scores', '00000001', 1376, 'mmstar')
    with pytest.raises(ValueError):
        numbered_filename('../1365')


def test_multimage_repeated_image_occurrences_and_nonsquare_grid():
    ids = [9] + [4] * 6 + [9, 10] + [4] * 2 + [9] + [4] * 6
    layout = image_layout(ids, [[1, 4, 6], [1, 2, 4], [1, 4, 6]], 4, 2, ['a', 'b', 'a'])
    assert [(x['grid_h'], x['grid_w']) for x in layout] == [(2, 3), (1, 2), (2, 3)]
    assert [(x['token_start'], x['token_end']) for x in layout] == [(1, 7), (9, 11), (12, 18)]
    with pytest.raises(ValueError, match='token count'):
        image_layout([9, 4, 4, 4], [[1, 4, 4]], 4, 2, ['a'])
    with pytest.raises(ValueError, match='still images'):
        image_layout([4] * 8, [[2, 4, 4]], 4, 2, ['a'])


def test_overlay_preserves_zero_and_places_hotspot_without_transpose():
    original = Image.new('RGB', (120, 80), (50, 60, 70))
    weights = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32)
    actual = np.asarray(overlay_image(original, weights, 1, alpha=1))
    assert tuple(actual[0, -1]) == (255, 0, 0)
    assert tuple(actual[-1, 0]) == (50, 60, 70)
    np.testing.assert_array_equal(np.asarray(overlay_image(original, weights * 0, 0)), np.asarray(original))


def test_export_global_scale_raw_mass_and_script_escaping(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (120, 80), (30, 60, 90)).save(source)
    out = tmp_path / 'out'; out.mkdir(); (out / 'attention').mkdir()
    frames = []
    for i, value in enumerate([0.2, 0.02], 1):
        raw = f'attention/latent_{i:05d}.npz'
        np.savez_compressed(out / raw, image_001=np.array([[[0, value], [0, 0]]], dtype=np.float32))
        frames.append({'ordinal': i, 'segment': 1, 'latent_step': i, 'decode_step': i,
                       'context': '', 'raw_attention': raw})
    manifest = {'record_file': '00001365.json', 'sample_id': '1376', 'layers': [0],
                'question': '</script><script>alert(1)</script>', 'saved_text': 'answer', 'replay_note': '',
                'policy_mismatches': [], 'images': [{'image': 1, 'path': str(source), 'grid_h': 2, 'grid_w': 2}],
                'frames': frames}
    render_outputs(out, manifest, alpha=.65, max_side=256)
    saved = json.loads((out / 'manifest.json').read_text(encoding='utf8'))
    a, b = [f['images'][0] for f in saved['frames']]
    assert a['vmax'] == b['vmax'] == pytest.approx(.2)
    assert a['attention_mass'] == pytest.approx(.2) and b['attention_mass'] == pytest.approx(.02)
    with Image.open(out / a['overlay']) as first, Image.open(out / b['overlay']) as second:
        assert np.asarray(first)[0, -1, 0] > np.asarray(second)[0, -1, 0]
    assert (out / 'contact_image_001_page_001.jpg').is_file()
    assert '</script><script>alert(1)' not in (out / 'index.html').read_text(encoding='utf8')


def test_layer_selection():
    assert select_layers([-2, -1], 4) == [2, 3]
    assert select_layers('all', 2) == [0, 1]
    with pytest.raises(ValueError):
        select_layers([-4], 2)
