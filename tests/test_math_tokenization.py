# SPDX-License-Identifier: MIT
"""Exercise the delivered tokenizer without requiring a GPU or modifying model assets."""

import os

import pytest

from scripts.prepare_runtime import native_prompt_builder


def test_math_filter_uses_native_deepseek_prompt_encoder():
    model = os.environ.get("RLK_RECIPE_TEST_MODEL")
    if not model:
        pytest.skip("set RLK_RECIPE_TEST_MODEL to the locked delivered model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    assert tokenizer.chat_template is None
    messages = [{"role": "user", "content": "What is 2 + 2?"}]
    with pytest.raises(ValueError, match="chat_template"):
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    builder = native_prompt_builder(model, tokenizer)
    assert type(builder).__name__ == "DeepSeekV4ContinuousTokenBuilder"
    # Golden single-user non-thinking render from the locked native/reference encoder.
    expected = tokenizer.encode(
        "<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2?<｜Assistant｜></think>",
        add_special_tokens=False,
    )
    assert builder.build_initial_tokens(messages) == expected
    assert len(expected) > 0
