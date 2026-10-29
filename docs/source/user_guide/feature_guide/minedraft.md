# MineDraft 并行投机解码（V0）

四数据集、AR/Standard SD/MineDraft 对比、k 扫描及性能汇总见 [四数据集复现实验](minedraft_datasets.md)。

此分支把 MineDraft 的批次交错调度接入 Ascend NPU：一张专用草稿卡生成候选 token，其余卡组成目标模型 TP 组，交错执行两组请求的草稿生成和验证。共享调度、缓存回退和拒绝采样算法仍位于 MineDraft 包，vllm-ascend 提供 NPU 执行与 HCCL 通信适配。

本实现面向 **vLLM 0.9.2 + vllm-ascend 0.9.2rc1 + 本分支 MineDraft**。本地尚未进行 NPU 实机验收；吞吐收益与模型、批量、接受率及通信开销有关，需在目标机器测量。

## 支持范围

- Linux Ascend 环境，V0 引擎（`VLLM_USE_V1=0`），eager 执行，单机 `mp`。
- 纯文本 dense 自回归目标模型和独立草稿模型；两者须使用兼容的 tokenizer、词表及 token ID。
- 目标 TP 可以为 1 或更大，草稿 TP 固定为 1，PP 和 DP 均为 1。
- 通过 batch expansion 验证候选，不启用 FlashAttention 专属的 MQA scorer。
- 不再要求安装 `torch-scatter`。

本适配不支持 V1、图执行、EAGLE/MTP/Medusa/MLP/ngram 草稿、MoE/EP、多模态、LoRA、分块预填充、前缀缓存、KV 传输、自定义 worker 和 Tetris。使用 `rejection_sampler`，并保持 `disable_logprobs=True`（默认值）；当前跨卡结果传输不支持完整的输出 logprobs。Tetris 会产生变长候选，而当前 batch expansion 路径尚未完成与之对应的映射适配，因此配置检查会拒绝启用。先使用下面的最小配置完成正确性验证，再扩大负载。

## 安装本地源码

按该版本的[安装文档](../../installation.md)准备匹配的驱动、CANN、PyTorch 和 torch-npu。将三个本地源码目录放入同一台 Linux NPU 机器，安装包含本次修改的代码，不能仅使用已发布的 MineDraft 或 vllm-ascend wheel。

下面假设目录为 `/workspace/vllm`、`/workspace/vllm-ascend`、`/workspace/MineDraft`，替换成实际位置：

```bash
cd /workspace/vllm
VLLM_TARGET_DEVICE=empty python -m pip install -e .

cd /workspace/vllm-ascend
python -m pip install -e . --no-build-isolation

cd /workspace/MineDraft
python -m pip install -e . --no-build-isolation

python -m pip check
```

vllm-ascend 的编译要求与原仓库一致。MineDraft 的 NPU 路径不依赖 CUDA、FlashInfer 或 `torch-scatter`；无需安装原 README 中的 NVIDIA 性能监控工具。

## 卡数与启动参数

**直接使用 `LLM` 或 `vllm serve` 时，`tensor_parallel_size` 包含草稿卡。**

| 配置 | API 的 `tensor_parallel_size` | 可见 NPU 数量 | 示例的 `--target-tp` |
| --- | --- | --- | --- |
| 草稿 1 卡 + 目标 1 卡 | 2 | 2 | 1 |
| 草稿 1 卡 + 目标 2 卡 | 3 | 3 | 2 |
| 草稿 1 卡 + 目标 4 卡 | 5 | 5 | 4 |

可见设备列表中的第一张卡是草稿卡，后续卡是目标模型卡。目标模型的注意力头数须能被**目标 TP** 整除。例如 `ASCEND_RT_VISIBLE_DEVICES=4,0,1,2,3` 表示物理卡 4 为草稿，物理卡 0–3 为目标。

MineDraft 会将调度器的请求容量扩展为两组交错批次。`max_num_seqs` 是传入模型执行配置的批量上限，不能把额外草稿卡计入目标模型分片数。

最小配置如下，环境变量须在导入 vLLM 前设置：

```python
import os

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ASCEND_ENABLE_MINEDRAFT"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_PLUGINS"] = "ascend,ascend_enhanced_model,minedraft"

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/models/target",
        tensor_parallel_size=2,  # 目标 TP=1，加一张草稿卡
        distributed_executor_backend="mp",
        enforce_eager=True,
        disable_async_output_proc=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        max_num_seqs=4,
        max_model_len=1024,
        gpu_memory_utilization=0.65,
        speculative_config={
            "model": "/models/draft",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 3,
            "is_parallel": True,
            "disable_mqa_scorer": True,
        },
    )
    try:
        outputs = llm.generate(
            ["The capital of France is", "Explain photosynthesis:"],
            SamplingParams(temperature=0, max_tokens=32),
        )
        for output in outputs:
            print(output.outputs[0].text)
    finally:
        llm.llm_engine.model_executor.shutdown()


if __name__ == "__main__":
    main()
```

## 实机正确性验收

示例 `examples/offline_inference_minedraft.py` 默认先运行不加载 MineDraft 的目标模型基线，再运行并行投机解码，逐请求比较 greedy token ID。两个模式使用不同进程，避免全局补丁及通信组残留；基线使用与并行模式目标模型相同的物理卡。示例自动在并行子进程设置 `VLLM_ASCEND_ENABLE_MINEDRAFT=1`，基线设置为 `0`。直接使用 API 时需要手动设置该开关；未启用时保持普通 Ascend 行为。

```bash
cd /workspace/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1
python examples/offline_inference_minedraft.py \
  --model /models/target \
  --draft-model /models/draft \
  --target-tp 1 \
  --num-speculative-tokens 3 \
  --max-num-seqs 4 \
  --rounds 2 \
  --timeout 1800 \
  --results-dir minedraft-results/tp1
```

示例覆盖多轮 `generate()`、单请求长解码、长短混合输入、超过两组容量的排队，以及输出长度不同导致的提前结束。成功时打印 `PASS` 并返回 0；token 不一致、子进程错误或超时返回非零。超时范围包含模型加载、生成和退出；父进程会终止该模式的进程组，避免遗留等待通信的 worker。请使用预先下载好的模型路径，下载时间也会计入超时。

输出目录保留 `baseline.json` 和 `parallel.json`，包括每个用例的 token ID、文本和生成耗时。生成耗时包含首次执行，不是正式性能基准。若比较失败，应检查首个分歧 token、词表/tokenizer 和目标模型精度；不同批次形状的浮点数差异也可能影响 greedy 结果，不应直接把差异当成通过。

扩大验证时分别执行：

1. 目标 TP=2（可见三卡，`--target-tp 2`），确认目标 TP 通信不包含草稿 rank。
2. `--num-speculative-tokens 1` 和更大的 k，检查首轮、复用和回退。
3. `--max-num-seqs 1`，检查很小批量及尾部请求能结束。
4. 提高请求数和输出长度，检查多轮执行后的显存、退出及预期输出长度。

可用 `--mode parallel` 单独运行并行模式；`--mode baseline` 单独保存目标基线。单独模式仍沿用“第一张可见卡预留给草稿”的卡号约定，便于比较同一组目标卡。

随机采样需要独立的分布正确性验证，不能要求两次随机采样生成完全相同的文本。当前示例不代替在线并发、取消请求、内存抢占和性能压测。

## Qwen3 TP=4 离线性能指标

`examples/offline_inference_minedraft_qwen3_tp4.py` 使用 Qwen3-32B 作为目标模型（TP=4），Qwen3-0.6B 作为草稿模型（TP=1），共需五张 NPU。默认预热一轮、正式运行三轮，每轮八个请求。可增加请求数和输出长度：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,0,1,2,3
export TARGET_MODEL=/models/Qwen3-32B
export DRAFT_MODEL=/models/Qwen3-0.6B
python examples/offline_inference_minedraft_qwen3_tp4.py \
  --warmup-runs 1 \
  --runs 3 \
  --num-prompts 32 \
  --max-tokens 256 \
  --max-num-seqs 4 \
  --output minedraft-results/qwen3-tp4.json
```

每轮和汇总均输出实际输入/输出 token 数、推理耗时、输出 token/s、总 token/s、请求/s，以及请求完成延迟 E2E、首 token 延迟 TTFT、每请求平均 TPOT 和首次调度排队时间的 mean/p50/p95/p99。JSON 另外保留配置、各轮逐请求数据和采样器计数差值。`--print-outputs` 可打印最后一轮回答；`--ignore-eos` 可用于固定输出长度测试。

计时仅覆盖 `generate()`，包含请求提交、排队、prefill、decode 和输出处理，排除模型加载、外部 chat template 分词、预热、打印和指标读取。汇总吞吐量为各轮实际 token 总数除以各轮推理总耗时。模型及引擎初始化时间单独记录。

有效草稿接受率为“首次拒绝前接受的草稿前缀 token 数 / 参与验证的草稿 token 数”，使用正式运行前后的累计计数差值，不含预热，也不包含未进入验证的草稿；验证计数可能包含最终因 EOS 或长度上限被截断的 token。TTFT/E2E 使用引擎请求时间戳并包含排队；TPOT 为每请求 `(完成时间 - 首 token 时间) / (输出 token 数 - 1)`，不代表投机解码批量返回 token 时的逐 token 间隔分布。缺失指标显示 `N/A`，JSON 中保存 `null` 或不可用原因。

这是重复八条内置问题的离线工作负载；少量样本的延迟百分位仅供观察。该脚本没有运行关闭 MineDraft 的目标模型基线，因此不报告加速比。

## Qwen3 TP=4 baseline 性能测试

`examples/offline_inference_baseline_qwen3_tp4.py` 是独立脚本，导入 vLLM 前设置 `VLLM_ASCEND_ENABLE_MINEDRAFT=0`，并将插件白名单限定为 Ascend 插件。每次须在新 Python 进程中运行。它提供两种模式：

- `--mode sd`（默认）：Standard SD，目标 Qwen3-32B TP=4，草稿 Qwen3-0.6B TP=1，与目标共享四张 NPU。这对应 MineDraft README 和 `scripts/experiment_1_sequential.sh` 使用的主要对照方案。
- `--mode ar`：仅目标 Qwen3-32B TP=4，关闭投机解码，用于补充比较。

可沿用 MineDraft 的五卡列表；脚本会跳过第一张专用草稿卡，仅使用后四张目标卡。也可直接传入四张目标卡。下面的命令使用与上一节相同的请求参数：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,0,1,2,3
export TARGET_MODEL=/models/Qwen3-32B
export DRAFT_MODEL=/models/Qwen3-0.6B
python examples/offline_inference_baseline_qwen3_tp4.py \
  --mode sd \
  --warmup-runs 1 --runs 3 --num-prompts 32 \
  --max-tokens 256 --max-num-seqs 4 --num-speculative-tokens 5 \
  --output minedraft-results/baseline-sd-qwen3-tp4.json

python examples/offline_inference_baseline_qwen3_tp4.py \
  --mode ar \
  --warmup-runs 1 --runs 3 --num-prompts 32 \
  --max-tokens 256 --max-num-seqs 4 \
  --output minedraft-results/baseline-ar-qwen3-tp4.json
```

预热、计时范围、吞吐量和请求延迟统计与 MineDraft 性能脚本一致。SD 同时报告逐位置接受率和有效前缀接受率；前者包含首次拒绝后的独立通过位置。后者基于原生 vLLM 0.9.2 固定 k 路径，由 `采样器输出 token 数 - 验证草稿 token 数 / k` 得到接受前缀数，再除以验证草稿数。计数不含预热，可能包含最终因停止条件而被截断的 token。AR 的草稿指标为不可用。

对比时使用相同目标物理卡、模型精度、请求数、输出长度、采样参数和 EOS 选项；需要固定长度时，两份脚本都添加 `--ignore-eos`。这里沿用前一性能示例的 `temperature=0` 和内置问题，并未复现原项目完整的数据集及随机采样实验。MineDraft 使用五卡，baseline 使用四卡；MineDraft 还会扩展两组交错调度容量，因此相同 `max-num-seqs` 不表示相同总调度容量。
