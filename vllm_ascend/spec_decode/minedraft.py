# SPDX-License-Identifier: Apache-2.0
"""Backend hooks for the optional MineDraft V0 plugin.

This module deliberately has no imports of MineDraft or device libraries at
module scope, so ordinary inference does not depend on the plugin.
"""


def is_minedraft_parallel(vllm_config) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return getattr(speculative_config, "is_parallel", False) is True


def get_minedraft_model_parallel_config(vllm_config):
    """Use the model's TP size for head counts and cache memory accounting.

    The target worker's parallel config describes all processes, including
    the drafter. Use a shallow copy so adjusting a model shape never changes
    the process topology or the config shared with the worker's runner.
    """
    parallel = vllm_config.parallel_config
    if (is_minedraft_parallel(vllm_config)
            and vllm_config.model_config.runner_type != "draft"):
        from copy import copy

        parallel = copy(parallel)
        # world_size remains the complete process count even while MineDraft
        # temporarily adjusts tensor_parallel_size during cache allocation.
        parallel.tensor_parallel_size = parallel.world_size - 1
    return parallel


def validate_minedraft_config(vllm_config) -> None:
    """Fail before launching workers for unsupported parallel configurations.

    MineDraft's tensor_parallel_size includes the dedicated drafter: rank zero
    drafts and ranks 1..N run a target model with tensor parallel size N.
    """
    if not is_minedraft_parallel(vllm_config):
        return

    from vllm import envs

    def require(condition, message):
        if not condition:
            raise ValueError("MineDraft on Ascend: " + message)

    spec = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    draft = getattr(spec, "draft_model_config", None)
    require(not envs.VLLM_USE_V1, "set VLLM_USE_V1=0 (V0 only).")
    require(model is not None and draft is not None,
            "a target model and an independent draft model are required.")
    require(parallel.tensor_parallel_size >= 2,
            "tensor_parallel_size must equal target TP + 1 (at least 2).")
    require(parallel.pipeline_parallel_size == 1
            and parallel.data_parallel_size == 1,
            "pipeline and data parallel sizes must both be 1.")
    require(parallel.distributed_executor_backend == "mp",
            "use the single-node multiprocessing backend "
            "(distributed_executor_backend='mp').")
    require(parallel.worker_cls in
            ("auto", "vllm.spec_decode.spec_decode_worker.create_spec_worker")
            and parallel.sd_worker_cls in
            ("auto", "vllm_ascend.worker.worker.NPUWorker"),
            "custom worker classes are not supported.")
    require(not parallel.enable_expert_parallel,
            "expert parallelism is not supported.")
    require(spec.draft_tensor_parallel_size == 1,
            "draft_tensor_parallel_size must be 1.")
    require(model.get_vocab_size() == draft.get_vocab_size(),
            "target and draft must have the same vocabulary size and token IDs.")
    require(getattr(spec, "method", None) in (None, "draft_model")
            and not getattr(spec, "prompt_lookup_max", 0),
            "only independent draft models are supported; n-gram, EAGLE, "
            "Medusa, MLP speculators and MTP are not supported.")
    require(getattr(spec, "acceptance_method", "rejection_sampler")
            == "rejection_sampler", "use rejection_sampler acceptance.")
    require(not getattr(spec, "force_mqa", False),
            "force_mqa is unsupported; Ascend uses batch expansion scoring.")
    require(spec.disable_logprobs,
            "disable_logprobs must be True for parallel result transport.")
    require(not getattr(spec, "tetris", False),
            "Tetris with variable proposal lengths is not supported.")

    for name, config in (("target", model), ("draft", draft)):
        require(config.enforce_eager,
                name + " model must use enforce_eager=True.")
        expected_runner = "generate" if name == "target" else "draft"
        require(config.runner_type == expected_runner
                and not getattr(config, "is_encoder_decoder", False)
                and not getattr(config, "is_multimodal_model", False)
                and not getattr(config, "is_attention_free", False)
                and not getattr(config, "has_inner_state", False),
                name + " must be a text-only, decoder-only attention model.")
        hf_config = getattr(config, "hf_text_config", config.hf_config)
        require(getattr(hf_config, "model_type", "") not in
                ("eagle", "eagle3", "medusa", "mlp_speculator", "deepseek_mtp"),
                name + " must be an independent autoregressive model.")
        expert_fields = ("moe_num_experts", "num_experts", "n_routed_experts",
                         "num_local_experts")
        require(not any((getattr(hf_config, field, 0) or 0) > 0
                        for field in expert_fields),
                name + " MoE models are not supported.")

    scheduler = vllm_config.scheduler_config
    require(not getattr(scheduler, "chunked_prefill_enabled", False),
            "disable chunked prefill.")
    require(not getattr(scheduler, "is_multi_step", False),
            "multi-step scheduling is not supported.")
    require(getattr(vllm_config, "lora_config", None) is None
            and getattr(vllm_config, "prompt_adapter_config", None) is None,
            "LoRA and prompt adapters are not supported.")
    require(getattr(vllm_config, "kv_transfer_config", None) is None,
            "KV transfer is not supported.")
    require(not getattr(vllm_config.cache_config, "enable_prefix_caching", False),
            "disable prefix caching.")
    additional = vllm_config.additional_config or {}
    require(not additional.get("expert_tensor_parallel_size", 0),
            "do not override expert_tensor_parallel_size.")
    require(not additional.get("torchair_graph_config", {}).get("enabled", False),
            "Torchair graph mode is not supported.")
    require(not additional.get("ascend_scheduler_config", {}).get("enabled", False),
            "the Ascend scheduler is not supported.")
    spec.disable_mqa_scorer = True


def get_minedraft_draft_model_runner_cls(vllm_config=None):
    """Return the NPU runner while retaining its CPU multi-step fallback."""
    from vllm_ascend.worker.draft_model_runner import TP1DraftModelRunner

    if vllm_config is not None:
        # The draft worker has its own TP=1 context, never the target's ETP.
        vllm_config.parallel_config.expert_parallel_size = 1
        vllm_config.parallel_config.expert_tensor_parallel_size = 1
    return TP1DraftModelRunner
