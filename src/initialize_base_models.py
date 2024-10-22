import argparse

from transformers import Speech2TextFeatureExtractor

from huggingface_asr.src.models.decoders.multi_head_gpt2 import (
    GPT2MultiHeadConfig
)
from huggingface_asr.src.models.encoders.e_branchformer import (
    Wav2Vec2EBranchformerConfig,
)


def init_feature_extractor(name):
    config = {
        "do_ceptral_normalize": True,
        "feature_size": 80,
        "normalize_means": True,
        "normalize_vars": True,
        "num_mel_bins": 80,
        "padding_side": "right",
        "padding_value": 0.0,
        "return_attention_mask": True,
        "sampling_rate": 16000
    }
    feature_extractor = Speech2TextFeatureExtractor.from_dict(config)
    feature_extractor.push_to_hub(name)


def init_small_encoder(name):
    conf = {
        "activation_dropout": 0.1,
        "add_adapter": False,
        "apply_spec_augment": False,
        "attention_dropout": 0.1,
        "conv_bias": False,
        "conv_dim": [
            256,
            256
        ],
        "conv_kernel": [
            3,
            3
        ],
        "conv_stride": [
            2,
            2
        ],
        "csgu_activation": "identity",
        "csgu_conv_dropout": 0.1,
        "csgu_kernel_size": 31,
        "csgu_use_linear_after_conv": False,
        "diversity_loss_weight": 0.1,
        "ebranchformer_conv_dropout": 0.1,
        "eos_token_id": 2,
        "fe_position_embeddings": False,
        "feat_extract_activation": "gelu",
        "feat_extract_norm": "group",
        "feat_proj_dropout": 0.0,
        "feat_quantizer_dropout": 0.0,
        "final_dropout": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-05,
        "layerdrop": 0,
        "max_source_positions": 1024,
        "merge_conv_kernel": 31,
        "num_attention_heads": 4,
        "num_conv_pos_embedding_groups": 16,
        "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 2,
        "num_hidden_layers": 12,
        "num_mel_bins": 80,
        "output_hidden_size": 256,
        "position_embeddings_type": "relative",
        "proj_codevector_dim": 256,
        "rotary_embedding_base": 10000,
        "use_fbanks": True,
        "use_macaron_ff": True,
    }

    configuration = Wav2Vec2EBranchformerConfig(**conf)
    configuration.push_to_hub(name)


def init_small_decoder(name):
    conf = {
        "activation_function": "gelu_new",
        "attn_pdrop": 0.1,
        "bos_token_id": 50256,
        "embd_pdrop": 0.1,
        "eos_token_id": 50256,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "n_embd": 256,
        "n_head": 4,
        "n_inner": 2048,
        "n_layer": 6,
        "n_positions": 1024,
        "output_hidden_size": 256,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.1,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 5000,
        "head_locations": [],
        "head_weights": [1.0],
    }
    configuration = GPT2MultiHeadConfig(**conf)
    configuration.push_to_hub(name)


def init_base_encoder(name):
    conf = {
        "activation_dropout": 0.1,
        "add_adapter": False,
        "apply_spec_augment": False,
        "attention_dropout": 0.1,
        "conv_bias": False,
        "conv_dim": [
            512,
            512
        ],
        "conv_kernel": [
            3,
            3
        ],
        "conv_stride": [
            2,
            2
        ],
        "csgu_activation": "identity",
        "csgu_conv_dropout": 0.1,
        "csgu_kernel_size": 31,
        "csgu_use_linear_after_conv": False,
        "diversity_loss_weight": 0.1,
        "ebranchformer_conv_dropout": 0.1,
        "eos_token_id": 2,
        "fe_position_embeddings": False,
        "feat_extract_activation": "gelu",
        "feat_extract_norm": "group",
        "feat_proj_dropout": 0.0,
        "feat_quantizer_dropout": 0.0,
        "final_dropout": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout": 0.1,
        "hidden_size": 512,
        "initializer_range": 0.02,
        "intermediate_size": 2048,
        "layer_norm_eps": 1e-05,
        "layerdrop": 0,
        "max_source_positions": 1024,
        "merge_conv_kernel": 31,
        "num_attention_heads": 4,
        "num_conv_pos_embedding_groups": 16,
        "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 2,
        "num_hidden_layers": 16,
        "num_mel_bins": 80,
        "output_hidden_size": 256,
        "position_embeddings_type": "relative",
        "proj_codevector_dim": 256,
        "rotary_embedding_base": 10000,
        "use_fbanks": True,
        "use_macaron_ff": True,
    }

    # Wav2vec2 base like model
    configuration = Wav2Vec2EBranchformerConfig(**conf)
    configuration.push_to_hub(name)


def init_base_decoder(name):
    conf = {
        "activation_function": "gelu_new",
        "attn_pdrop": 0.1,
        "bos_token_id": 0,
        "embd_pdrop": 0.1,
        "eos_token_id": 1,
        "head_locations": [
        ],
        "head_weights": [
            1.0
        ],
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "n_embd": 512,
        "n_head": 8,
        "n_inner": 2048,
        "n_layer": 8,
        "n_positions": 1024,
        "reorder_and_upcast_attn": False,
        "resid_pdrop": 0.1,
        "scale_attn_by_inverse_layer_idx": False,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "tie_additional_weights": False,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 500
    }

    configuration = GPT2MultiHeadConfig(**conf)
    configuration.push_to_hub(name)


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize base models")
    parser.add_argument("--feature_extractor_name", type=str, help="Name of feature extractor to push to the hub")
    parser.add_argument("--small_encoder_name", type=str, help="Name of small encoder to push to the hub")
    parser.add_argument("--small_decoder_name", type=str, help="Name of small decoder to push to the hub")
    parser.add_argument("--base_encoder_name", type=str, help="Name of base encoder to push to the hub")
    parser.add_argument("--base_decoder_name", type=str, help="Name of small decoder to push to the hub")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_feature_extractor(args.feature_extractor_name)
    init_small_encoder(args.small_encoder_name)
    init_small_decoder(args.small_decoder_name)
    init_base_encoder(args.base_encoder_name)
    init_base_decoder(args.base_decoder_name)
