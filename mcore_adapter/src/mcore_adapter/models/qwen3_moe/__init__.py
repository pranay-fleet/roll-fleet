from ..auto.config_auto import register_config
from ..auto.modeling_auto import register_model
from ..converter.dist_converter import default_dist_config, register_dist_config, shared_moe_dist_config
from ..converter.template import (
    QKVBiasConverOp,
    QKVConverOp,
    RenameConverOp,
    StackConverOp,
    register_template,
)
from ..model_config import McaModelConfig
from ..model_factory import McaGPTModel


register_config("qwen3_moe", McaModelConfig)
register_model("qwen3_moe", McaGPTModel)
register_dist_config("qwen3_moe", default_dist_config.merge_configs(shared_moe_dist_config))


register_template(
    "qwen3_moe",
    hf_layer_prefix="model.layers.",
    hf_moe_prefix=".mlp.experts.",
    config_hf_to_mca={
        "max_position_embeddings": "max_sequence_length",
        "hidden_size": "hidden_size",
        "attention_bias": "add_qkv_bias",
        "head_dim": "kv_channels",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_query_groups",
        "num_hidden_layers": "num_layers",
        "rms_norm_eps": "layernorm_epsilon",
        "vocab_size": "padded_vocab_size",
        "attention_dropout": "attention_dropout",
        "rope_theta": "rotary_base",
        "intermediate_size": "ffn_hidden_size",
        "tie_word_embeddings": "tie_embeddings_and_output_weights",
        # MoE related
        "moe_intermediate_size": "moe_ffn_hidden_size",
        "decoder_sparse_step": "moe_layer_freq",
        "num_experts": "num_moe_experts",
        "num_experts_per_tok": "moe_router_topk",
        "router_aux_loss_coef": "moe_aux_loss_coeff",
    },
    constant_mca_config={
        "swiglu": True,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "add_bias_linear": False,
        "hidden_dropout": 0.0,
        "rotary_percent": 1.0,
        "moe_router_load_balancing_type": "aux_loss",
        "moe_router_pre_softmax": False,
        "qk_layernorm": True,
    },
    weight_converters=[
        RenameConverOp(hf_names="lm_head.weight", mca_names="output_layer.weight"),
        RenameConverOp(hf_names="model.embed_tokens.weight", mca_names="embedding.word_embeddings.weight"),
        RenameConverOp(hf_names=".input_layernorm.weight", mca_names=".self_attention.linear_qkv.layer_norm_weight"),
        RenameConverOp(hf_names=".self_attn.o_proj.weight", mca_names=".self_attention.linear_proj.weight"),
        RenameConverOp(hf_names=".self_attn.q_norm.weight", mca_names=".self_attention.q_layernorm.weight"),
        RenameConverOp(hf_names=".self_attn.k_norm.weight", mca_names=".self_attention.k_layernorm.weight"),
        RenameConverOp(hf_names=".post_attention_layernorm.weight", mca_names=".pre_mlp_layernorm.weight"),
        RenameConverOp(hf_names=".down_proj.weight", mca_names=".linear_fc2.weight"),
        RenameConverOp(hf_names="model.norm.weight", mca_names="decoder.final_layernorm.weight"),
        StackConverOp(hf_names=[".gate_proj.weight", ".up_proj.weight"], mca_names=".linear_fc1.weight", dim=0),
        RenameConverOp(hf_names=".mlp.gate.weight", mca_names=".mlp.router.weight"),
        QKVConverOp(
            hf_names=[".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight"],
            mca_names=".self_attention.linear_qkv.weight",
        ),
        QKVBiasConverOp(
            hf_names=[".self_attn.q_proj.bias", ".self_attn.k_proj.bias", ".self_attn.v_proj.bias"],
            mca_names=".self_attention.linear_qkv.bias",
        ),
    ],
)
