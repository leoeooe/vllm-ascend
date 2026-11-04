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

| 引擎 | 驱动逻辑设备 ID | TP | 职责 |
| --- | --- | --- | --- |
| 草稿 | 0 | 1 | 为所有 batch 生成确定性草稿 |
| 验证副本 0 | 1, 2 | 2 | 持有分配给自己的请求及 target KV cache |
| 验证副本 1 | 3, 4 | 2 | 持有其余请求及 target KV cache |

请求首先均匀分为 `num_batches` 个 batch，batch 固定分配到验证副本。草稿引擎轮流处理已得到验证结果的 batch。一个请求只有当前轮已经验证完成，才会向草稿引擎提交下一轮已确认前缀；版本检查防止使用旧前缀的草稿。

设备参数是每个引擎子进程要写入 `ASCEND_RT_VISIBLE_DEVICES` 的驱动逻辑设备 ID，不是外层可见设备列表中的相对下标。双芯片卡的物理卡号与这些 ID 不同，请用 `npu-smi info` 核对映射，并只使用当前容器实际分配的设备。每个子进程在导入模型运行时前设置设备可见性，避免草稿引擎加入验证模型的 TP/HCCL 组。

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

需要最多五个可用 NPU 设备和 batch-invariant 算子；条件不足时测试跳过。

### 2026-11-04 实卡验证

使用 Ascend 910、CANN 9.2.1、torch 2.10.0 和 torch_npu 2.10.0.post7，通过上面的示例入口执行了以下功能测试。每项包含 8 条中英文短请求，每条生成 24 个 token；`K=3`、`max_model_len=1024`、每个 worker 的 KV cache 为 1 GiB。精确比较开启 `VLLM_BATCH_INVARIANT=1`，普通 target 基线先在独立进程中生成并退出。

| 草稿 / target | TP2 验证副本 | batch | 模式 | 输出校验 | 草稿接受数 | 图重放：草稿 / 各验证 rank |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B / 0.6B | 2 | 4 | eager | 与已校验的图模式 192 个 token 全同 | 141/148 | 0 / 0, 0, 0, 0 |
| 0.6B / 0.6B | 2 | 4 | 图 | 与普通 target 的 192 个 token 全同 | 141/148 | 50 / 12, 12, 12, 12 |
| 0.6B / 0.6B | 1 | 2 | 图 | 与普通 target 的 192 个 token 全同 | 141/148 | 26 / 12, 12 |
| 0.6B / 32B | 2 | 4 | 图 | 与普通 target 的 192 个 token 全同 | 111/219 | 83 / 21, 21, 15, 15 |
| 0.6B / 32B | 2 | 4 | eager | 与普通 target 的 192 个 token 全同 | 111/219 | 0 / 0, 0, 0, 0 |

模型为本地 Qwen3-0.6B 和 Qwen3-32B。验证副本固定使用驱动逻辑设备 `8,9` 和 `10,11`；最初草稿使用 `6`，该设备被另一任务占用后迁至 `12`，单副本测试使用 `12` 和 `8,9`。所有验证 rank 均使用 `ExternalSpeculator`，没有本地草稿模型；图模式逐 rank 检查了实际 replay，未仅以捕图成功作为通过条件。

加载前面的 CANN 和 `PYTHONPATH` 环境后，32B 图模式的复现命令为（运行前重新确认设备空闲）：

```bash
VLLM_BATCH_INVARIANT=1 /home/l00998189/vllm/.venv/bin/python \
    examples/minedraft_npu.py \
    --target /home/d00832329/Qwen3-32B \
    --draft /home/d00832329/Qwen3-0.6B \
    --draft-devices 12 --verifier-devices '8,9;10,11' \
    --num-batches 4 --num-speculative-tokens 3 --max-tokens 24 \
    --kv-cache-memory-bytes 1073741824 --check-baseline \
    --report minedraft-32b-graph.json
```

实测期间保留了三次失败记录：CANN `PYTHONPATH` 未保留导致 `acl` 导入失败；自动估算约 20 GiB KV cache 时首次 FIA prefill OOM；逻辑设备 `6` 被其他任务占用约 55 GiB 后，大模型拓扑的草稿引擎无法启动。补齐环境、显式使用 1 GiB KV cache、迁移草稿设备后，上表测试通过。自动容量配置下的 OOM 根因尚未确认，不能把短请求测试通过视为大 KV cache 配置已修复。

正常关闭的外层等待窗口已在 vLLM 提交 `135770c955` 中修正；异常路径仍执行有界进程组终止。修正后的标准 pytest 结果为 **9 passed、26 subtests passed**，实卡运行不再出现此前的共享内存泄漏告警。Ascend 测试入口的显式 KV cache 参数在提交 `ee8fddd94` 中保存。

原始 JSON、日志、`summary.json` 汇总和设备快照保存在开发机 `/home/l00998189/runtime/minedraft-validation/20261104/`。32B 的 eager/图模式输出也相互一致，单/双验证副本的小模型输出相互一致；最终设备快照确认测试进程已释放 NPU。这些结果验证短请求 greedy 正确性、批次切换、非对称副本和解码图重放；尚未验证长上下文压力、随机采样分布或吞吐提升。主机时间线中的 drafting/verification 重叠也不等价于 NPU kernel 重叠。

## 小体积图模式 profiling

`examples/minedraft_npu.py --profile-dir DIR` 会先完成加载、图捕获及一次不采集的预热，再向全部引擎广播 `start_profile`，等待所有 worker 确认后运行短生成，最后广播 `stop_profile`。各引擎步数不同，因此不使用相同的 `max_iterations` 截断各自时间窗。采集默认只包含 **4 条请求 × 8 个输出 token**，预热也为每条 8 个 token；这不是吞吐基准。

该入口通过 `profiler_kwargs={"timing_only": True}` 选择 Ascend 的轻量设置：仅 NPU activity、`Level0`、`AiCoreNone`、Text 导出，关闭 CPU 算子、stack、modules、shapes、memory、FLOPS、L2 和算力指标。profiler 按需创建，避免未启动对象析构干扰正在采集的会话。TP worker 是 daemon 进程，采集时关闭自动解析；模型全部退出后，由主进程串行离线解析，解析时间不计入采集窗口。

在前面的 CANN、`PYTHONPATH` 和 HCCL 环境中执行：

```bash
VLLM_BATCH_INVARIANT=1 /home/l00998189/vllm/.venv/bin/python \
    examples/minedraft_npu.py \
    --target /home/d00832329/Qwen3-32B \
    --draft /home/d00832329/Qwen3-0.6B \
    --draft-devices 14 --verifier-devices '4,5;6,7' \
    --num-batches 4 --num-prompts 4 --max-tokens 8 \
    --profile-warmup-tokens 8 --num-speculative-tokens 3 \
    --kv-cache-memory-bytes 1073741824 \
    --profile-dir /home/l00998189/runtime/minedraft-profile/new-capture \
    --report /home/l00998189/runtime/minedraft-profile/new-report.json
```

运行前确认设备仍空闲，并使用空的 `--profile-dir`，避免混入旧采集。草稿使用逻辑设备 `14`（物理卡 7），两个验证副本分别使用 `4,5`（物理卡 2）和 `6,7`（物理卡 3）。`--num-prompts`、`--max-tokens` 控制采集体积；`--profile-warmup-tokens` 只影响不采集的预热。未提供 `--profile-dir` 时，示例保持原来的 8 请求、32 输出 token 默认配置。

输出包括：

- `mindstudio_timing.zip`：优先下载的小包，包含合并 NPU 时间线、manifest 和生成诊断。
- `minedraft_timeline.json`：只保留各 worker 原生设备任务的完整时间事件和泳道元数据，角色名称为 `draft.tp0.npu14`、`verifier0.tp0.npu4` 等。
- `native_timelines.zip`：可选的原生 worker 时间线包，不修改原始事件。
- `native/*_ascend_pt/`：服务器上保留的原生采集和解析文件。

下载并解压小包后，在 MindStudio Insight 中选择“导入数据”，导入 `minedraft_timeline.json`，然后打开 Timeline，按角色展开 NPU stream，框选任务范围或点击任务查看时长。官方支持本地 JSON 文件及目录导入，参见 [MindStudio 基础操作](https://github.com/Ascend/msinsight/blob/master/docs/zh/user_guide/basic_operations.md)。

合并仅隔离不同 worker 的 PID，并增加角色标签；原始 `ts`、`dur` 和设备任务参数保持不变，**不把各卡首事件分别归零**。CPU/flow 事件不进入精简视图；原生文件仍保留在可选包中。查看并行关系时不要手动把不同卡的起点对齐；MindStudio 的时间偏移行为可参考[时间对齐说明](https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0053.html)。合并文件用于时间线查看，不等价于原生集群工程的通信矩阵或算子汇总；需要这些视图时使用原生数据。NPU 任务时间不包含 Python 调度与进程通信开销，主机调用区间另外保存在生成报告的 `timeline` 中。

2026-11-04 的精简实测文件位于 `/home/l00998189/runtime/minedraft-profile/20261104/capture-timing/`：主 ZIP 为 **4.59 MiB**，解压后的合并 JSON 为 **80.16 MiB**，包含 328,275 条任务及元数据事件。生成窗口为 749.93 ms；采集期间的图重放次数为草稿 `20`、验证副本 0 各 rank `5`、验证副本 1 各 rank `4`。32 个输出 token 与未采集的预热结果相同。

原始设备 `KERNEL_AICORE`/`KERNEL_AIVEC` 区间经并集去重后，草稿与任一验证副本的交集为 **67.58 ms**；这是本次短窗口存在设备计算重叠的证据，不是吞吐提升或请求时延估计。验收结果保存在同级父目录的 `timing-validation.json`，全部测试设备已释放。代码检查通过，Ascend profiler/合并工具 CPU 测试为 **23 passed**，vLLM 控制接口标准 pytest 为 **11 passed、32 subtests passed**。本机没有运行 MindStudio GUI，已检查 JSON 结构、原始时间戳、五个 worker 的设备任务和压缩包完整性；GUI 展示仍需在本地导入确认。

## GSM8K 固定长度运行实验

示例支持通过 `--dataset-path` 读取本地 GSM8K JSONL，并通过 `--dataset-offset` 指定从哪条记录开始，默认从第 0 条开始。`--num-prompts` 指定实际读取的记录数；不会通过重复内置 prompt 补齐数量。输入采用原始 completion 模板 `Question: {question}\nAnswer:`，不添加 chat template、few-shot 示例或参考答案。报告保留数据文件 SHA256、记录索引、问题、参考答案、生成文本和 token 数，便于复核。这是运行与调度实验，不是标准 GSM8K 准确率评测。

2026-11-04 从 [GSM8K 官方仓库](https://github.com/openai/grade-school-math) 的提交 `3101c7d5072418e28b9008a6636bde82a006892c` 下载了原始 train/test 数据，保存到 `/home/l00998189/datasets/gsm8k/`。其中 `train.jsonl` 有 7,473 条记录，`test.jsonl` 有 1,319 条；同目录保存许可证、上游说明和下载校验清单 `manifest.json`。测试集 SHA256 为 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。

本实验使用测试集前 32 条，分成 4 个 batch，每批 8 条；`K=5`，每条生成 128 token。沿用示例的 `ignore_eos=True`，用于固定长度负载。采用 `VLLM_BATCH_INVARIANT=0`，开启默认 prefix cache；这与此前用于逐 token 对照的 batch-invariant 配置不同。预热生成 8 token 后单独记录正式生成的时间线、图重放及投机计数增量，避免把预热计入测量。

每次完整验证需要 `8 * (5 + 1) = 48` 个查询 token。示例根据 batch 请求数和 K 补充图捕获尺寸，避免原有上限 32 使正常验证回退 eager。`--max-num-batched-tokens 4096` 独立于单请求 `--max-model-len 1024`，为八条请求的 prefill 留出预算。每个 worker 显式使用 2 GiB KV cache；设备及容量须在运行前检查。

在前述 CANN、`PYTHONPATH` 和 HCCL 环境中，从 Ascend 仓库运行：

```bash
VLLM_BATCH_INVARIANT=0 /home/l00998189/vllm/.venv/bin/python \
    examples/minedraft_npu.py \
    --target /home/d00832329/Qwen3-32B \
    --draft /home/d00832329/Qwen3-0.6B \
    --draft-devices 14 --verifier-devices '10,11;12,13' \
    --dataset-path /home/l00998189/datasets/gsm8k/test.jsonl \
    --dataset-offset 0 --num-prompts 32 --num-batches 4 \
    --num-speculative-tokens 5 --max-tokens 128 \
    --max-model-len 1024 --max-num-batched-tokens 4096 \
    --kv-cache-memory-bytes 2147483648 --warmup-tokens 8 \
    --report /home/l00998189/runtime/minedraft-gsm8k/20261104/report.json
```

该命令保存主机调用时间线，不生成整段 4,096 输出 token 的 NPU trace。主机调用交叠不能直接换算为设备计算交叠。`--profile-dir` 仍会采集整个正式生成，若添加该参数，文件体积会明显大于此前 4 条、每条 8 token 的精简采样。

首次尝试沿用逻辑 `4,5;6,7` 作为验证设备。启动前只见每卡约 113 MiB 的背景上下文，随后另一组 `vLLM-Omni` 任务占用了这些设备，初始化时每卡仅余 2.72–3.20 GiB，未能加载验证模型。本次进程退出后，经用户确认，改用草稿 `14`、验证副本 `10,11` 和 `12,13`，保留原先的 TP2×2 拓扑。首次失败日志和前后设备快照保存在实验目录中，没有终止其他任务。

迁移后的第一次完整生成得到 32×128 个输出 token，图重放检查通过，但新增测量代码错误地保留了预热诊断中共享的 `timeline` 列表，导致正式生成后切片得到空时间线。提交 `b33fb377d` 在生成前冻结诊断快照，并新增共享列表回归测试；修复后重新执行相同配置。缺失时间线的报告保留为 `attempt2-timeline-bug-report.json`，不用于逐轮耗时结论。数据集入口和图尺寸支持保存在前一提交 `b5312c44b`。

### GSM8K 实测结果（2026-11-04）

最终运行通过，32 条请求均输出 128 token，共 **4,096 token**。正式生成耗时 **7.8882 s**，包含该次请求的 prefill，不包含模型加载、图捕获、8-token 预热和退出；对应本次负载的输出速率为 519.25 token/s。它是单次运行的描述性结果，没有同条件串行 SD 或普通 target 性能基线，不能据此计算加速比。参考答案未参与输入，也没有将截断到 128 token 的输出计为标准 GSM8K 准确率。

正式时间线保留 **290** 项调用：4 次 target prefill、143 次 draft、143 次 verify。为比较完整 batch 的供给速度，选取每批第 **3–21 轮**：初始 prefill 产生 1 token，每轮最多确认 K+1=6 token，因此第 21 轮之前每请求至少仍剩 7 token；这一范围保证 8 条请求仍活跃且候选长度为 5，同时跳过最初两轮的流水线填充。这里统计的是 actor 完整调用耗时，包含主机及同步开销。

| 引擎 | 采样调用数 | 中位耗时 | P90 耗时 | `2 × median(Tdraft) / median(Tverify)` |
| --- | --- | --- | --- | --- |
| 草稿，逻辑 14 | 76 | 53.46 ms | 54.83 ms | — |
| verifier0，逻辑 10、11 | 38 | 44.28 ms | 45.09 ms | 2.415 |
| verifier1，逻辑 12、13 | 38 | 43.96 ms | 44.40 ms | 2.432 |

此配置仍未满足 `Tdraft <= Tverify / 2`：验证耗时约 44 ms 时，草稿需要约 22 ms/批或更低，而实测约 53 ms/批。增加到每批 8 请求、K=5 后，草稿仍是供给瓶颈；不能通过较大的主机重叠比例认定两个 verifier 已被充分供满。

采集增量确认草稿实际图重放 **565** 次，verifier0 两个 rank 各 **57** 次，verifier1 两个 rank 各 **60** 次。目标捕获尺寸为 `[6,12,18,36,48]`，包含完整 batch 需要的 48 token；短尾/变长情况仍可能回退 eager。原生 speculative counters 中，两个副本分别接受 **1533/2524（60.74%）**、**1516/2606（58.17%）** 个草稿 token，这不是数学答案准确率。各引擎累计 KV preemption 为 0。

两个 verifier 的 verify 调用分别有 **94.20%**、**99.17%** 的时间与 draft 调用交叠；草稿与任一 verify 的主机调用交集为 **6.315 s**。这些区间包含主机工作，不能与此前 NPU profiler 的 67.58 ms 设备计算交集直接比较。本次其他物理卡有并发任务，且请求、batch 大小、K、batch invariance 和采集设置均与此前短采样不同。

结果与失败记录位于 `/home/l00998189/runtime/minedraft-gsm8k/20261104/`：`report.json` 保存逐题输出和原始诊断，`analysis.json` 保存完整 batch 统计，`input-capacity.json` 保存输入长度及 KV 容量核算，`run.sh`/`run.log` 保存复现命令及日志，`after-npu.txt` 确认测试设备已释放。修复前后两次完整运行的 4,096 个输出 token 相同，但未额外执行普通 target 正确性基线。新增 CPU 测试 **8 passed**，代码检查通过。

## 当前范围与后续研究

- 支持固定 K、多个 batch、独立 TP 验证副本、原生 target 拒绝采样及每个引擎的解码图。单引擎内部使用同步调度，跨引擎由控制器并发驱动。
- 初始入口只覆盖离线文本、`n=1`；不覆盖 OpenAI 在线服务、流式输出、多模态、结构化输出、PP、CP 或引擎内 DP。多验证副本由控制器管理。
- 验证引擎必须保留原生 scheduler；Ascend 的 recompute、dyntra、profiling chunk、balance 和其他自定义 scheduler 会在配置阶段被拒绝，避免绕过外部 batch 选择。
- 草稿侧每轮通过已确认前缀调用普通生成引擎，可利用 prefix cache；没有新增持久化草稿 KV 状态回滚协议。重复 prefill、CPU token IPC 和负载不均衡都可能抵消重叠收益。
- JSON 中的时间区间表示进程调用的并发，包含 CPU、通信和同步开销。使用 NPU profiler 确认设备 kernel 重叠，再比较固定请求集上的吞吐、TTFT、TPOT、接受率和总设备成本。
- 后续创新可围绕持久化草稿 KV、动态 batch 分配、每副本队列反压、K 自适应以及设备间 token 通信展开。修改时保留版本号、已验证前缀和图缓冲区所有权约束。
