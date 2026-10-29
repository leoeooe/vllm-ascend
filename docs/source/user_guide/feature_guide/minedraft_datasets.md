# MineDraft 四数据集复现实验（Ascend）

入口是 `examples/benchmark_minedraft_datasets.py`，与同目录的 `minedraft_dataset_io.py`、`minedraft_dataset_worker.py` 一起使用。支持 Chatbot Arena（用户指定的 FastChat 数据）、GSM8K、ShareGPT 和 Spec-Bench，自动运行 AR、Standard SD 和 MineDraft，并生成性能与正确性报告。

这是 vLLM 0.9.2、vllm-ascend 0.9.2rc1 和本分支 MineDraft 的 NPU 复现流程。当前已知多请求 queued greedy 对比存在待定位差异；本脚本保留检查，不能以完成性能测量代替正确性通过。新增脚本仅完成静态检查，尚未在开发机执行 NPU 推理。

## 数据与实验范围

| 参数名 | 数据来源 | 用法 |
| --- | --- | --- |
| `fastchat` | [LMSYS Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | 保留 conversation_a/b 双侧的第一条用户输入，与 MineDraft 转换脚本一致 |
| `gsm8k` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)，main/test | 使用 question，不将 answer 放入输入 |
| `sharegpt` | [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) 中 ShareGPT_V3_unfiltered_cleaned_split.json | 第一条用户输入 |
| `spec_bench` | [Spec-Bench question.jsonl](https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl) | turns[0]，保留 category |

[FastChat](https://github.com/lm-sys/FastChat) 是项目名称；此入口固定使用 Arena，不使用 MT-Bench。这里测量首轮生成性能，不执行官方 MT-Bench judge、GSM8K 答案准确率或 Spec-Bench 全部任务评分。

原 MineDraft 的四份输入是 ShareGPT、Arena、Spec-Bench 和 tough。这里按用户要求将 tough 替换为 GSM8K，因此属于指定四数据集的 Ascend 复现与扩展，不应声称复现了原论文的全部结果。PEARL、Tetris、EAGLE 等其他方法不在这个脚本的运行范围内。

默认参数参照 MineDraft `scripts/experiment_1_sequential.sh` 与 `experiment_1_parallel.sh`：

| 参数 | 默认值 |
| --- | --- |
| 目标 / 草稿 | Qwen3-32B / Qwen3-0.6B |
| 目标 TP / 草稿 TP | 4 / 1 |
| k | 1、2、3、4、5 |
| 每数据集请求数上限 | 1000 |
| 每请求输出长度 | 256，忽略 EOS |
| temperature / top_p / n | 1.0 / 1.0 / 1 |
| warmup / 正式轮数 | 1 / 3 |
| max_num_seqs / max_model_len | 16 / 2048 |
| 精度 / 显存比例 | BF16 / 0.65 |
| prompt | 原始用户文本，不加 chat template |
| 最小输入 token 数 | Arena 16、GSM8K 4、ShareGPT 256、Spec-Bench 4 |

请求超过长度预算会被过滤，既不截断也不重复凑数。实际样本数、过滤数量、来源 SHA256、选中 ID、最终输入 token 和总 fingerprint 都保存在 `prompts.json`。数据集内容自然重复（例如 Arena 双侧同一问题）会保留。

## 环境与数据准备

先按 [MineDraft 安装说明](minedraft.md) 安装本分支三个项目。在已有匹配环境中补充数据下载依赖：

```bash
python -m pip install datasets
```

已有本地 JSON/JSONL 文件时不需要 `datasets` 下载依赖。默认 `--data-dir` 下识别这些文件：

```text
minedraft-data/
  arena.jsonl                 # 也接受 arena.json，包括原 MineDraft 转换结果
  gsm8k_test.jsonl            # 也接受 gsm8k.json / gsm8k.jsonl
  ShareGPT.json               # 也接受 sharegpt.json / sharegpt.jsonl
  spec_bench.jsonl            # 也接受 spec_bench.json
```

可以用 `--fastchat-path`、`--gsm8k-path`、`--sharegpt-path`、`--spec-bench-path` 分别指定其他位置。GSM8K 支持 question/answer，聊天数据支持 conversations 或 messages；Spec-Bench 支持 turns。无效行、非文本或图像输入会跳过并计数。

缺少源文件时，只有显式添加 `--download-data` 才会下载。Arena 的 Hugging Face 数据访问条件需事先通过账号接受；下载使用环境中已有的登录信息，不会自动同意数据条款。如果服务器离线，可以在联网机器准备上述原始文件再传入路径。下载缓存同时写入 `.source.json` 来源信息，复用时验证 SHA256；修改了缓存文件应使用新的数据目录，避免混淆来源。

仅准备数据、固定样本及分词，不加载模型权重：

```bash
cd /workspace/vllm-ascend
python examples/benchmark_minedraft_datasets.py \
  --model /models/Qwen3-32B \
  --data-dir /datasets/minedraft \
  --download-data --prepare-only \
  --output-dir minedraft-results/prepared
```

分词使用目标模型 tokenizer；`--prompt-format chat` 可切换为 chat template 并关闭 Qwen3 thinking。对比三种模式时必须使用相同输入格式。`--min-input-tokens 0` 可覆盖最短输入限制。

## 在 NPU 上运行

设备列表始终按“专用草稿卡、目标卡”排列。下面物理卡 4 为 MineDraft 的草稿卡，物理卡 0–3 为目标；AR/SD 只使用物理卡 0–3。Standard SD 的草稿 TP=1 与目标共用其中一张卡。

建议先用较少样本确认实机路径：

```bash
python examples/benchmark_minedraft_datasets.py \
  --model /models/Qwen3-32B \
  --draft-model /models/Qwen3-0.6B \
  --devices 4,0,1,2,3 --target-tp 4 --dtype bfloat16 \
  --data-dir /datasets/minedraft \
  --datasets fastchat gsm8k sharegpt spec_bench \
  --modes ar sd minedraft --k-values 5 \
  --num-prompts 32 --max-num-seqs 4 \
  --max-tokens 256 --warmup-runs 1 --runs 3 \
  --output-dir minedraft-results/qwen3-small
```

完整 k 扫描：

```bash
python examples/benchmark_minedraft_datasets.py \
  --model /models/Qwen3-32B \
  --draft-model /models/Qwen3-0.6B \
  --devices 4,0,1,2,3 --target-tp 4 --dtype bfloat16 \
  --data-dir /datasets/minedraft \
  --modes ar sd minedraft --k-values 1 2 3 4 5 \
  --num-prompts 1000 --max-num-seqs 16 \
  --max-tokens 256 --warmup-runs 1 --runs 3 \
  --temperature 1.0 --gpu-memory-utilization 0.65 \
  --output-dir minedraft-results/qwen3-full
```

每个模式/k/阶段使用独立进程以隔离全局插件；每个进程顺序处理四个数据集，模型只加载一次。默认完整扫描为 11 个正确性进程和 11 个性能进程。逐进程日志会实时写入输出目录；`--timeout` 默认 7200 秒，包含该进程的模型加载、所有数据集和退出。超时/异常会终止进程组并保留已有结果。输出目录必须为空或尚不存在，避免覆盖旧实验。

复用已固定输入时，加 `--bundle /path/to/prepared/prompts.json`；`--model`、数据集列表、请求数、seed、输入格式、长度限制等准备参数必须与该文件一致。对照额外草稿模型时，可分别将 `--draft-model` 改为 Qwen3-1.7B / Qwen3-4B，使用同一 bundle 和不同输出目录；批量扫描也应分别保存。

## 正确性检查与已有 queued 问题

默认先以 AR 为参照，对每种待测模式/k 执行独立的 greedy 检查。它使用相同的输入 token、`temperature=0`、`ignore_eos=True`，覆盖 single 和长短混合 queued 请求。默认 queued 请求数为 `2 * max_num_seqs + 1`，因此能覆盖超出两组交错容量后的补入；实际数据不足时会明确记录 `passed_limited_coverage`，不会伪造更多样本。

`correctness.json` 保存所有已比较请求的 mismatch、首次不同 token 下标（从 0 开始）以及两侧 token ID；对应 worker JSON 保存完整生成文本。任意 mismatch 默认阻止性能阶段，退出码为 2。之前 TP=4 的 queued 差异并未因新增 benchmark 自动修复，若再次出现应保留这些诊断结果继续定位。

可选 `--continue-on-mismatch` 在差异后继续测性能，但最终报告仍保持失败状态，退出码仍为 2。`--skip-correctness` 只测性能，结果明确标记 `not_run` / `completed_unvalidated`。这两个选项用于测量或诊断，均不会声明正确复现。

性能阶段默认 temperature=1.0，因此不要求随机输出逐 token 相同；独立 greedy 检查通过也不代表已经证明随机采样分布完全等价。若只想复现前面的 greedy 性能配置，可设置 `--temperature 0 --prompt-format chat`；输出按实际 token 数计量，`--respect-eos` 则允许提前停止。

## 结果与统计口径

输出目录主要包含：

- `prompts.json`：统一输入、样本 ID、参考答案（不进入 prompt）、源 SHA256、过滤统计及 fingerprint。
- `correctness.json`：greedy 对照结论、所有差异、排队检查覆盖情况。
- `performance-<mode>-k<k>.json`：每数据集/每轮/每请求的耗时、token ID、生成文本和指标；同时记录软件版本、配置与初始化时间。
- `*.job.json`、`*.log`：逐进程配置及实时输出；正确性阶段也有对应 JSON。
- `summary.json`、`summary.csv`、`summary.txt`：完整实验状态和汇总表。失败时仍保留已完成部分。

指标包括实际输入/输出 token 数、总推理耗时、输出 token/s、输入加输出总 token/s、request/s；E2E、TTFT、每请求平均 TPOT、首次调度排队时间的 mean/p50/p95/p99；以及有效草稿接受率。延迟缺失时为 `null`，并保留有效样本数。

计时仅覆盖 `generate()`，包括请求提交、排队、prefill、decode 和输出处理，排除模型初始化、外部分词、预热、指标读取与文件写入。吞吐量使用所有正式轮次的实际 token 总数除以推理耗时总和。TTFT 是引擎请求时间，不是网络客户端时间；TPOT 是每请求 `(完成时间-首 token 时间)/(输出 token 数-1)` 的均值，不能当成投机解码逐 token 间隔。

接受率读取预热后采样器累计计数差值。MineDraft 使用有效接受前缀/已验证草稿 token；Standard SD 基于原生 vLLM 0.9.2 固定 k 计数推导有效前缀，另保留逐位置接受率。采样器计数可能包含最终因 EOS/长度上限被截断的 token，所以吞吐量始终使用真正返回的 token 数。

加速比基于同一数据集的输出吞吐量，报告 `speedup_vs_ar`、`speedup_vs_sd_same_k` 与 `speedup_vs_best_measured_sd`；最后一项仅针对本次实测 k 集合的最佳 SD。MineDraft 比 SD 多一张专用草稿卡，并扩展两组调度容量。报告同时保留正确性状态；未检查或失败的测量不能作为通过正确性验收的加速结果。此脚本不会在没有实机数据时生成论文数值或宣称收益达标。
