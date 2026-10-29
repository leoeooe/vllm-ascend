# SPDX-License-Identifier: Apache-2.0
"""NPU smoke run: Qwen3-32B target TP=4 + Qwen3-0.6B draft TP=1.

Requires five visible NPUs and the modified MineDraft/vllm-ascend branches.
Set TARGET_MODEL and DRAFT_MODEL to local model directories if needed.
"""

import os

# Configure the platform and plugins before importing vLLM or torch.
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4")
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ASCEND_ENABLE_MINEDRAFT"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_PLUGINS"] = "ascend,ascend_enhanced_model,minedraft"


def main():
    devices = [value.strip() for value in
               os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
    if len(devices) != 5 or len(set(devices)) != 5 or not all(devices):
        raise ValueError("Select exactly five distinct NPUs: draft 1 + target 4")

    from vllm import LLM, SamplingParams

    print(f"Draft NPU: {devices[0]}; target TP=4 NPUs: {devices[1:]}",
          flush=True)
    llm = LLM(
        model=os.environ.get("TARGET_MODEL", "Qwen/Qwen3-32B"),
        # MineDraft's process count includes the dedicated draft rank.
        tensor_parallel_size=5,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        disable_async_output_proc=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.70,
        speculative_config={
            "model": os.environ.get("DRAFT_MODEL", "Qwen/Qwen3-0.6B"),
            "method": "draft_model",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 5,
            "is_parallel": True,
            "force_pearl": False,
            "tetris": False,
            "disable_mqa_scorer": True,
            "disable_logprobs": True,
            "acceptance_method": "rejection_sampler",
        },
    )
    try:
        questions = [
            "用三句话解释什么是机器学习。",
            "解释大语言模型中 KV Cache 的作用。",
            "解释为什么投机解码可以加速推理。",
            "用一个例子解释张量并行。",
            "写一个 Python 函数判断整数是否为素数。",
            "比较 CPU 和 GPU 各自适合的工作。",
            "介绍二分查找的步骤和时间复杂度。",
            "用通俗语言解释 Transformer 的注意力机制。",
        ]
        tokenizer = llm.get_tokenizer()
        prompts = [
            {"prompt_token_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )}
            for question in questions
        ]
        outputs = llm.generate(
            prompts, SamplingParams(temperature=0, max_tokens=128),
            use_tqdm=True)
        for question, output in zip(questions, outputs):
            completion = output.outputs[0]
            print(f"\n问题：{question}\n回答：{completion.text}")
            print(f"生成 token 数：{len(completion.token_ids)}")
        print(f"\n完成 {len(outputs)} 个请求。")
    finally:
        llm.llm_engine.model_executor.shutdown()


if __name__ == "__main__":
    main()
