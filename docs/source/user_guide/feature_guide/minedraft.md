# MineDraft 多批次投机解码（实验性）

本分支实现独立草稿引擎与验证引擎之间的多批次流水线。一个 batch 等待草稿时，验证引擎可以运行另一个已准备好的 batch。草稿引擎和各验证副本拥有独立进程、设备组、KV cache 和图捕获缓冲区。

实现参考 [MineDraft 论文](https://arxiv.org/abs/2603.18016) 的批次间 drafting/verification 重叠机制，扩展为可配置的 batch 数量和非对称模型副本数量。这里的“双流”指两个阶段并发推进；不同 NPU 上的计算流属于各自进程，不是在同一进程中复制两个 `torch.npu.Stream`。

## 版本与入口

开发基线：

- vLLM Ascend main：`3b715ff78066d22b95be6f4a84e861a44930b25a`。
- 对应的 vLLM main：`24420b7a341af37a9bc3186e6fc6062512e0aa58`，来自 Ascend 的 `.github/vllm-main-verified.commit`。
- 两个仓库的开发分支：`feat/minedraft-multibatch`。必须安装两个仓库的开发分支；仅升级 Ascend 不会添加上游的 `MineDraftLLM`、外部草稿注入和调度器接口。

公开入口为 `vllm.entrypoints.minedraft.MineDraftLLM`，目前用于离线、文本、token ID 输入。它返回按原始 prompt 顺序排列的标准 `RequestOutput`。

## 五卡非对称部署

| 引擎 | 物理设备 ID | TP | 职责 |
| --- | --- | --- | --- |
| 草稿 | 0 | 1 | 为所有 batch 生成确定性草稿 |
| 验证副本 0 | 1, 2 | 2 | 持有分配给自己的请求及 target KV cache |
| 验证副本 1 | 3, 4 | 2 | 持有其余请求及 target KV cache |

请求首先均匀分为 `num_batches` 个 batch，batch 固定分配到验证副本。草稿引擎轮流处理已得到验证结果的 batch。一个请求只有当前轮已经验证完成，才会向草稿引擎提交下一轮已确认前缀；版本检查防止使用旧前缀的草稿。

设备参数是每个引擎子进程要写入 `ASCEND_RT_VISIBLE_DEVICES` 的物理 ID，不是外层可见设备列表中的相对下标。请只使用当前容器实际分配的设备。每个子进程在导入模型运行时前设置设备可见性，避免草稿引擎加入验证模型的 TP/HCCL 组。

在已安装匹配 CANN、torch、torch_npu 和两个源码分支的环境中，从 Ascend 仓库目录运行：

```bash
source /usr/local/Ascend/cann/set_env.sh
export PYTHONPATH=/home/l00998189/vllm:/home/l00998189/vllm-ascend:${PYTHONPATH:-}
export HCCL_NPU_SOCKET_PORT_RANGE=auto

/home/l00998189/vllm/.venv/bin/python examples/minedraft_npu.py \
    --target /home/d00832329/Qwen3-32B \
    --draft /home/d00832329/Qwen3-0.6B \
    --draft-devices 0 \
    --verifier-devices '1,2;3,4' \
    --num-batches 4 \
    --num-speculative-tokens 3 \
    --kv-cache-memory-bytes 1073741824 \
    --report minedraft-graph-report.json
```

该示例默认启用 `FULL_DECODE_ONLY`。添加 `--eager` 可运行无图基线。配置 `--verifier-devices '1,2' --num-batches 2` 可测试一个验证副本上的双 batch 流水线。`num_batches` 与验证副本数不同：增加 batch 不会重复加载模型。

图捕获会增加启动时间和显存占用。示例限制了捕获尺寸，但是否能在单卡/双卡放下指定模型，仍取决于权重类型、设备容量、上下文和 KV cache 配置。示例不加载量化配置，需使用适合设备容量的模型或通过 Python 接口传入相应的 vLLM 参数。

`--kv-cache-memory-bytes` 显式指定每个 worker 的 KV cache 字节数，优先于 `--gpu-memory-utilization` 的自动估算。上面的 1 GiB 用于示例的短请求验证；长上下文或更多并发请求需要重新计算容量。加载 CANN 环境后保留其 `PYTHONPATH`，否则 worker 可能无法导入 `acl`。

## Python 用法

```python
from vllm import SamplingParams
from vllm.entrypoints.minedraft import MineDraftLLM

if __name__ == "__main__":
    # 实际使用时，以 target tokenizer 将用户文本编码成 token ID。
    prompts = [[100, 200, 300], [100, 201, 301]]
    with MineDraftLLM(
        draft_model="Qwen/Qwen3-0.6B",
        target_model="Qwen/Qwen3-8B",
        draft_devices=[0],
        verifier_devices=[[1, 2], [3, 4]],
        num_batches=4,
        num_speculative_tokens=3,
        max_model_len=1024,
        enforce_eager=False,
        worker_extension_cls=(
            "vllm_ascend.worker.v2.minedraft.MineDraftWorkerExtension"
        ),
        probe_method="minedraft_probe",
    ) as llm:
        outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32))
        diagnostics = llm.get_diagnostics()
```

必须使用 `if __name__ == "__main__"`，因为引擎使用 spawn 启动独立进程。`with` 在正常完成或异常时清理引擎；也可显式调用 `close()`。`max_num_seqs`、`dtype`、`gpu_memory_utilization` 等普通 vLLM 参数可作为关键字传入，`draft_engine_kwargs` 可覆盖草稿侧参数。设备组、TP 数、投机方法和引擎内部调度方式由控制器管理。

## 图模式与正确性

每个引擎独立捕获和重放自己的模型图。草稿模型作为普通生成引擎运行，其单 token 解码使用现有 ACL 图；验证模型使用 `K+1` 查询长度的原生投机验证图。外部草稿 token ID 在图重放前写入稳定的输入缓冲区，IPC、batch 选择和变长结果处理留在图外。

这条路径不依赖 `StandaloneARSpeculator` 的图捕获实现；该实现在此 vLLM 基线上尚未支持图。验证端使用无本地草稿权重的 `ExternalSpeculator`，沿用 vLLM 原生 rejection sampler、纠正 token 和 KV 回退逻辑。

目前草稿采样固定为 greedy。target 可以采用 `temperature=0` 或普通随机采样：确定性草稿对应 one-hot 草稿分布，由原生拒绝采样处理。未实现跨进程传输完整草稿 logits 的 probabilistic drafter。

为了区分“已捕获”和“实际使用”，worker 扩展在工作负载前安装进程内计数器，诊断中分别报告 `captured_graphs` 和 `graph_replays`。该计数器只包裹图重放调用，不改变图内操作或全局类。示例在图模式下要求草稿及每个验证 rank 都有实际重放；也检查每个验证副本的投机轮数大于零。低接受率仍可能是正确结果，因此不同大小模型的示例不会强制接受率阈值。

## 硬件验证

严格的逐 token greedy 比较应开启 [batch invariance](batch_invariance.md)，避免不同批次形状产生的浮点差异。先安装对应的 batch-invariant 自定义算子，再运行：

```bash
VLLM_BATCH_INVARIANT=1 /home/l00998189/vllm/.venv/bin/python \
    examples/minedraft_npu.py \
    --target /home/d00832329/Qwen3-32B \
    --draft /home/d00832329/Qwen3-0.6B \
    --draft-devices 0 --verifier-devices '1,2;3,4' \
    --num-batches 4 --check-baseline \
    --report minedraft-correctness-report.json
```

普通 target 基线在独立子进程中运行并退出，之后才启动 MineDraft，避免占用相同 NPU。报告保存 token ID、逐 rank 图诊断、原生投机指标和各批次的主机时间区间。`--check-baseline` 失败时保留报告，不把差异当作测试通过。

硬件 e2e 覆盖一个 TP2 验证副本的两 batch 图模式，以及两个 TP2 验证副本的四 batch eager/图模式。使用相同的 Qwen3-0.6B 作为草稿和 target，以检查接受 token 路径：

```bash
/home/l00998189/vllm/.venv/bin/python -m pytest -v \
    tests/e2e/pull_request/eight_card/model_runner_v2/test_minedraft.py
```

需要最多五张可用 NPU 和 batch-invariant 算子；条件不足时测试跳过。开发沙箱没有暴露 `/dev/davinci*`，因此这些硬件测试尚未在此开发环境运行，不能据此声称 NPU 正确性、图重放或吞吐提升已经验证。

## 当前范围与后续研究

- 支持固定 K、多个 batch、独立 TP 验证副本、原生 target 拒绝采样及每个引擎的解码图。单引擎内部使用同步调度，跨引擎由控制器并发驱动。
- 初始入口只覆盖离线文本、`n=1`；不覆盖 OpenAI 在线服务、流式输出、多模态、结构化输出、PP、CP 或引擎内 DP。多验证副本由控制器管理。
- 验证引擎必须保留原生 scheduler；Ascend 的 recompute、dyntra、profiling chunk、balance 和其他自定义 scheduler 会在配置阶段被拒绝，避免绕过外部 batch 选择。
- 草稿侧每轮通过已确认前缀调用普通生成引擎，可利用 prefix cache；没有新增持久化草稿 KV 状态回滚协议。重复 prefill、CPU token IPC 和负载不均衡都可能抵消重叠收益。
- JSON 中的时间区间表示进程调用的并发，包含 CPU、通信和同步开销。使用 NPU profiler 确认设备 kernel 重叠，再比较固定请求集上的吞吐、TTFT、TPOT、接受率和总设备成本。
- 后续创新可围绕持久化草稿 KV、动态 batch 分配、每副本队列反压、K 自适应以及设备间 token 通信展开。修改时保留版本号、已验证前缀和图缓冲区所有权约束。
