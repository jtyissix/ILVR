"""Real tiny Qwen numerical tests: no downloaded weights and no GPU required."""
import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("qwen_vl_utils")

from eval.model import QwenBackbone


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (Qwen2TokenizerFast, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLImageProcessor)
    root = tmp_path_factory.mktemp("tiny_qwen") / "backbone"
    vocab = {"<unk>": 0, "<pad>": 1, "<|im_end|>": 2, "<|im_start|>": 3,
             "<|image_pad|>": 4, "<|vision_start|>": 5, "<|vision_end|>": 6,
             "<|latent_start|>": 7, "<|latent_end|>": 8,
             "user": 9, "assistant": 10, "Which": 11, "?": 12,
             "red": 13, "blue": 14, "Answer": 15, ":": 16}
    raw = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = Qwen2TokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
                                        eos_token="<|im_end|>", additional_special_tokens=list(vocab)[3:9])
    template = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                "{% for part in message['content'] %}{% if part['type'] == 'image' %}"
                "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ part['text'] }}{% endif %}{% endfor %}"
                "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    processor = Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(min_pixels=56*56, max_pixels=56*56),
                                   tokenizer=tokenizer, chat_template=template)
    config = Qwen2_5_VLConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=256, image_token_id=4, video_token_id=17,
                             vision_start_token_id=5, vision_end_token_id=6,
                             eos_token_id=2, pad_token_id=1, bos_token_id=3,
                             rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
                             vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                                            "num_heads": 4, "out_hidden_size": 32, "patch_size": 14,
                                            "temporal_patch_size": 2, "spatial_merge_size": 2,
                                            "window_size": 56, "fullatt_block_indexes": [0]})
    config.sft2_helper_conditioned_backbone = True
    config.latent_start_id, config.latent_end_id = 7, 8
    torch.manual_seed(42)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    model.save_pretrained(root)
    processor.save_pretrained(root)
    return root


@pytest.fixture(scope="module", params=["eager", "sdpa"])
def backend(checkpoint, request):
    return QwenBackbone(checkpoint.parent, device="cpu", dtype="float32", attention=request.param, max_new_steps=16)


def messages(tmp_path, with_image=True):
    from PIL import Image
    content = [{"type": "text", "text": "Which?"}]
    if with_image:
        path = tmp_path / "test.png"
        Image.new("RGB", (56, 56), "red").save(path)
        content.insert(0, {"type": "image", "image": str(path)})
    return [{"role": "user", "content": content}]


@pytest.mark.parametrize("with_image", [False, True])
def test_prefill_matches_native_qwen(backend, tmp_path, with_image):
    with torch.inference_mode():
        inputs = backend._inputs(messages(tmp_path, with_image))
        actual = backend._prefill(inputs)
        expected = backend.model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden = expected.hidden_states[-1] if isinstance(expected.hidden_states, tuple) else expected.hidden_states
        torch.testing.assert_close(actual["hidden"], hidden[:, -1:], atol=2e-6, rtol=2e-5)


def test_recurrent_cache_matches_full_sequence(backend, tmp_path):
    with torch.inference_mode():
        inputs = backend._inputs(messages(tmp_path))
        prefix, positions = backend._embed(inputs)
        state = backend._prefill(inputs)
        for token in [7, None, None, 8, 13]:
            embeds = state["hidden"] if token is None else backend.embedding(torch.tensor([[token]]))
            next_pos = torch.full((3, 1, 1), state["next_position"], dtype=torch.long)
            prefix = torch.cat([prefix, embeds], dim=1)
            positions = torch.cat([positions, next_pos], dim=-1)
            state = backend.advance(state, token)
            expected = backend.model.model(inputs_embeds=prefix, position_ids=positions,
                attention_mask=torch.ones((1, prefix.shape[1]), dtype=torch.long), use_cache=False, return_dict=True)
            torch.testing.assert_close(state["hidden"], expected.last_hidden_state[:, -1:], atol=2e-6, rtol=2e-5)


def test_candidate_nll_matches_masked_full_forward(backend, tmp_path):
    msg = messages(tmp_path)
    result = backend.likelihood(msg, {"A": "red", "B": "blue red"}, include_eos=True)
    with torch.inference_mode():
        inputs = backend._inputs(msg)
        prefix_len = inputs["input_ids"].shape[1]
        for key, answer in {"A": "red", "B": "blue red"}.items():
            tail = backend.tokenizer.encode(answer, add_special_tokens=False) + [backend.tokenizer.eos_token_id]
            full = copy.copy(inputs)
            full["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([tail])], dim=1)
            full["attention_mask"] = torch.ones_like(full["input_ids"])
            labels = full["input_ids"].clone()
            labels[:, :prefix_len] = -100
            native = backend.model(**full, labels=labels, use_cache=False, return_dict=True)
            assert result["nll"][key] == pytest.approx(native.loss.item(), abs=2e-6)
    reversed_result = backend.likelihood(msg, {"B": "blue red", "A": "red"}, include_eos=True)
    assert result["nll"] == pytest.approx(reversed_result["nll"], abs=2e-6)


def test_auto_loads_only_backbone_and_detects_token_mismatch(backend, checkpoint, tmp_path):
    assert backend.policy.enabled
    assert backend.runtime["loaded_modules"] == ["visual", "model", "lm_head"]
    assert not any("aux" in n or "interaction" in n or "cross_att" in n for n, _ in backend.model.named_modules())
    from eval.common import read_json, write_json
    import shutil
    copied = tmp_path / "bad"
    shutil.copytree(checkpoint, copied)
    config = read_json(copied / "config.json")
    config["latent_start_id"] = 18
    write_json(copied / "config.json", config)
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        QwenBackbone(copied, device="cpu", dtype="float32")


def test_generation_runs_for_base_and_latent_backbone(backend, checkpoint, tmp_path):
    assert "latent_segments" in backend.generate(messages(tmp_path))
    base = QwenBackbone(checkpoint, model_kind="base", device="cpu", dtype="float32", max_new_steps=3)
    result = base.generate(messages(tmp_path))
    assert result["latent_segments"] == [] and 7 not in result["token_ids"] and 8 not in result["token_ids"]
