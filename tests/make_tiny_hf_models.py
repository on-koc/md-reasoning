#!/usr/bin/env python
"""Create tiny, randomly initialised Hugging Face models for offline tests.

Produces ``<out>/tiny-chat-llm`` (Llama architecture with a ChatML template) and
``<out>/tiny-embedder`` (BERT architecture). Their outputs are meaningless; they
only exist to exercise the ``huggingface`` provider code paths without
downloading anything.

    python tests/make_tiny_hf_models.py --out /tmp/tiny-models
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
SPECIAL = ["<pad>", "<s>", "<unk>", "<|im_start|>", "<|im_end|>"]


def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    corpus = [
        "The Aurora Consortium monitors glaciers in the Kessler Range.",
        "Entities, relationships, communities and reports form a knowledge graph.",
        '{"named_entities": ["Radio City", "India"]} {"triples": [["a", "b", "c"]]}',
    ] * 50
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<pad>"
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.model_max_length = 4096
    return fast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import BertConfig, BertModel, GenerationConfig, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tok = build_tokenizer()

    llm_dir = args.out / "tiny-chat-llm"
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    model = LlamaForCausalLM(cfg)
    model.generation_config = GenerationConfig(
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
    )
    model.save_pretrained(llm_dir)
    tok.save_pretrained(llm_dir)

    emb_dir = args.out / "tiny-embedder"
    bert = BertModel(
        BertConfig(
            vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=64, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        )
    )
    bert.save_pretrained(emb_dir)
    emb_tok = build_tokenizer()
    emb_tok.model_max_length = 512
    emb_tok.save_pretrained(emb_dir)
    print(f"wrote {llm_dir} and {emb_dir}")


if __name__ == "__main__":
    main()
