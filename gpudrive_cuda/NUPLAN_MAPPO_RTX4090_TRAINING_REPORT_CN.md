# nuPlan 真实多场景 RTX 4090 MAPPO 训练运行报告

## 1. 报告摘要

本报告记录一次完整的真实 nuPlan 多场景 DangerMaker MAPPO 训练。流水线从工作站已有
的 nuPlan SQLite 和 HD Map 开始，确定性抽取 128 个候选 scene，转换为
CanonicalScenario 与固定容量 RuntimeScenario，建立 64 个训练场景和 16 个独立评估
场景，最终在单张 RTX 4090 上完成 500 万 world-step 的课程训练、checkpoint 保存、
确定性评估和 GIF/MP4 可视化。

本次运行成功结束：

```text
Git 基线                         67122f0
流水线启动                      2026-08-14 02:38:07 CST
流水线结束                      2026-08-14 21:52:04 CST
总墙钟时间                      约 19 小时 14 分钟
流水线退出码                    0
原始 scene                      1,364
候选 / Canonical / Runtime      128 / 128 / 128
初步合格 Canonical              123
train / eval                    64 / 16
train-eval scene token 重叠      0
目标 / 实际 world-step          5,000,000 / 5,005,312
PPO update                      611
最终训练 episode 成功率         0.98095
独立评估 ego 碰撞率             0.9375
最终 checkpoint                 已生成并可读取
GIF / MP4 / PNG                 已生成并验证
```

实际步数略高于 500 万是正常现象：训练按每个完整的 8,192 world-step rollout batch 更新，
最后一个完整 batch 使总数到达 5,005,312。

## 2. 结果解释与使用边界

独立评估显示，16 个 eval world 中 `ego_collision_rate=0.9375`，说明多场景
simulator-MAPPO 对抗闭环已经工作。但本次策略同时出现：

```text
offroad_rate             0.264151
non_ego_collision_rate   0.258503
```

因此当前策略仍包含明显的越界和无关车辆碰撞行为。该结果只能证明当前独立 simulator、
固定参考轨迹 ego 和 MAPPO 攻击策略之间的闭环有效，不能表述为攻破真实自动驾驶规划系统，
也不能仅引用 93.75% 的 ego 碰撞率而省略安全副作用。

## 3. 参考文档和执行入口

本次运行遵循：

```text
gpudrive_cuda/NUPLAN_BATCH_CONVERSION_AND_TRAINING.md
gpudrive_cuda/RTX4090_MAPPO_TRAINING_GUIDE.md
```

为保证长流水线在 SSH 断开后仍能自动执行，并在每个验收门槛失败时停止，本次增加：

```text
gpudrive_cuda/scripts/run_nuplan_batch_training.sh
```

该脚本依次执行：

```text
SQLite scene 统计与 seed=42 抽样
  -> 逐 scene CanonicalScenario 转换
  -> CanonicalScenario 验证与初筛
  -> 固定容量 RuntimeScenario 编译与验证
  -> 64/16 split 与 token 去重检查
  -> Torch CUDA bridge 编译和单元测试
  -> RTX 4090 MAPPO 训练
  -> 固定 eval split 评估和媒体渲染
```

后台启动方式为：

```bash
tmux new -s dangermaker
cd /home/user/whz/agent

export PATH=/usr/local/cuda-12.4/bin:/home/user/anaconda3/envs/gpudrive/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=0
export CMAKE_CUDA_ARCHITECTURES=89
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -o pipefail
./gpudrive_cuda/scripts/run_nuplan_batch_training.sh 2>&1 \
  | tee /data/user/agent_runs/nuplan_mappo_128/pipeline.log
```

## 4. 工作站环境

### 4.1 GPU 与 CUDA

```text
GPU                       4 x NVIDIA GeForce RTX 4090
训练卡                    物理 GPU 0
显存                      24 GiB
NVIDIA Driver             570.133.07
compute capability        8.9
编译架构                  sm_89
训练 nvcc                 /usr/local/cuda-12.4/bin/nvcc
CUDA Toolkit              12.4.131
```

工作站同时存在 CUDA 11.8 和 Conda CUDA 12.9。本次显式把 `/usr/local/cuda-12.4/bin`
放在 PATH 最前，确保扩展编译 CUDA 与 PyTorch wheel 的 CUDA 12.4 完全匹配。

### 4.2 Python 与训练依赖

```text
uv                        0.12.3
项目 Python               3.10.20
PyTorch                   2.6.0+cu124
torch.version.cuda        12.4
torch.cuda.is_available   True
```

nuPlan 转换环境：

```text
NUPLAN_PYTHON             /home/user/anaconda3/envs/gpudrive/bin/python
nuPlan devkit             /data/fc/hugsim/atg/nuplan-devkit
scenario_pipeline         /home/user/whz/agent/scenario_pipeline
```

## 5. 原始数据与固定抽样

### 5.1 数据路径

```text
SQLite 根目录
/home/user/whz/agent/data/dataset/nuPlan/raw/cache

HD Map 根目录
/home/user/whz/agent/data/dataset/nuPlan/maps

地图元数据
/home/user/whz/agent/data/dataset/nuPlan/maps/nuplan-maps-v1.0.json
```

只读统计得到：

```text
数据库文件                 64
包含 scene 的有效数据库    64
原始 scene                 1,364
单个 DB 最少 scene         5
单个 DB 最多 scene         29
唯一 scene token           1,364
```

### 5.2 候选场景选择

```text
随机种子                   42
候选上限                   128
实际候选                   128
选择文件                   /data/user/agent_runs/nuplan_mappo_128/selected_scenes.tsv
选择文件大小               14,720 bytes
```

选择文件逐行保存 SQLite 绝对路径和 scene token，没有复制或修改原始数据库。该文件是本次
数据选择的关键复现元数据。

## 6. CanonicalScenario 转换与筛选

输出：

```text
/data/user/agent_runs/nuplan_mappo_128/canonical
```

转换结果：

```text
选择候选                   128
成功                       128
失败                       0
Canonical 文件             128
唯一 source scene ID       128
map-ready                  128
初步合格                   123
磁盘占用                   约 1.2 GiB
```

初筛条件为：

- 恰好一个 ego；
- HD Map 非空；
- 按 runtime 编译器顺序保留最近 64 个 agent 后，初始时刻至少有两辆有效非 ego 车辆。

Canonical 验证汇总为 `succeeded=128, warned=24, failed=0`。警告没有导致 schema 或数据
验证失败；123 个初步合格场景足以建立后续 80 场景正式划分。

相关日志：

```text
/data/user/agent_runs/nuplan_mappo_128/convert.log
/data/user/agent_runs/nuplan_mappo_128/failed_scenes.tsv
/data/user/agent_runs/nuplan_mappo_128/validate_canonical.log
```

`failed_scenes.tsv` 为零行。

## 7. 固定容量 RuntimeScenario

Runtime 输出：

```text
/data/user/agent_runs/nuplan_mappo_128/runtime
```

统一容量：

```text
max_agents                 64
history_steps              11
max_future_steps           128
max_map_features           2,048
max_map_points             32,768
max_map_edges              8,192
max_traffic_lights         128
max_route_features         512
```

结果：

```text
RuntimeScenario            128
验证失败                   0
磁盘占用                   约 110 MiB
```

编译与验证汇总均为 `succeeded=128, warned=127, failed=0`。大量 warning 主要来自固定容量
裁剪，并不表示二进制 runtime 损坏；后续 split、CUDA bridge、训练和评估均正常读取了这些
runtime。

相关日志：

```text
/data/user/agent_runs/nuplan_mappo_128/compile_runtime.log
/data/user/agent_runs/nuplan_mappo_128/validate_runtime.log
```

## 8. 64/16 数据划分验收

划分文件：

```text
/home/user/whz/agent/dataset/nuplan/rl_runtime/runtime_split.json
```

最终验收：

```text
train                       64
train unique token          64
eval                        16
eval unique token           16
train/eval token overlap    0
runtime root                /data/user/agent_runs/nuplan_mappo_128/runtime
min_attackers               2
require_map                 true
```

SHA-256：

```text
83808eed22a12e1c6e004eda2d24e575f0421c6e04c298541f98cf9c15ae3b68
```

当前划分保证 scene token 不重叠，但不是按原始 log/数据库分组的严格 log-disjoint 划分。
因此该评估可称为 scene-disjoint eval，不能称为严格的跨日志泛化评估。

## 9. Torch CUDA bridge 构建修复与测试

### 9.1 发现的问题

远程初始版本在本工作站上暴露两个阻塞训练的问题：

1. CMake 错误选择系统 Python 3.13，而 uv 项目环境使用 Python 3.10，pybind11 因此找不到
   `python_add_library`；
2. 预编译 PyTorch wheel 和 simulator 静态库使用了不同的 libstdc++ ABI，扩展导入时出现
   `undefined symbol: discover_runtime_scenes`。

### 9.2 修复

修改：

```text
gpudrive_cuda/CMakeLists.txt
gpudrive_cuda/scripts/build_torch_extension.sh
```

修复内容：

- 显式查找 Python `Interpreter` 和 `Development.Module`；
- 显式传递 uv Python 3.10 可执行文件给 CMake；
- 通过 uv 锁定环境运行 CMake；
- 把 PyTorch 的 C++ ABI 编译标志传播到 runtime、simulator、CLI 和测试目标。

构建后扩展：

```text
/home/user/whz/agent/build/gpudrive_cuda_rl/python/
gpudrive_cuda_torch.cpython-310-x86_64-linux-gnu.so
```

### 9.3 测试

```text
Python/MAPPO + Torch bridge tests     7
通过                                7
失败                                0
跳过                                0
runtime loader                       PASS
CUDA simulator integration           PASS
```

关键点是 `test_torch_bridge` 实际执行并通过，没有因扩展导入失败而被 skip。

## 10. 正式训练配置

解析后的配置：

```text
/home/user/whz/agent/outputs/mappo_rtx4090/resolved_config.json
```

核心规格：

```text
device                      cuda:0
seed                        42
num_worlds                  64
max_attackers               16
rollout_steps               128
每次更新 world-step         8,192
total_world_steps           5,000,000
checkpoint_interval         250,000
learning_rate               3e-4
gamma                       0.99
GAE lambda                  0.95
PPO epochs                  4
PPO minibatches             8
clip coefficient            0.2
value clip                  0.2
entropy coefficient         0.01
value coefficient           0.5
target KL                   0.02
max gradient norm           0.5
```

奖励权重：

```text
ego_collision               +10.0
progress                    +0.3
risk                        +0.2
non_ego_collision           -2.0
offroad                     -1.0
road_collision              -0.5
action_delta                -0.02
steering                    -0.02
acceleration                -0.01
time                        -0.01
```

## 11. 课程训练过程

训练共记录 611 条 update 指标。各阶段边界与训练 episode 成功率：

| 阶段 | 攻击车 | 控制模式 | 指标记录 | world-step 范围 | 阶段首/末成功率 |
|---|---:|---|---:|---:|---:|
| residual_4 | 4 | RESIDUAL | 62 | 8,192–507,904 | 0.8047 → 0.8327 |
| residual_8 | 8 | RESIDUAL | 122 | 516,096–1,507,328 | 0.8308 → 0.8923 |
| residual_16 | 16 | RESIDUAL | 183 | 1,515,520–3,006,464 | 0.8923 → 0.9274 |
| direct_16 | 16 | DIRECT | 244 | 3,014,656–5,005,312 | 0.9387 → 0.9810 |

第一条指标：

```text
world-step                   8,192
episode success rate         0.80469
episode timeout rate         0.19531
explained variance           0.00561
approx KL                    0.00302
吞吐                         72.38 world-step/s
```

最后一条指标：

```text
world-step                   5,005,312
update                       611
stage                        direct_16
episode success rate         0.98095
episode timeout rate         0.01905
episode return mean          -1.43484
episode length mean          19.7071
episodes completed           420
explained variance           0.88935
approx KL                    0.00000128
clip fraction                0
entropy                      3.33223
gradient norm                66.9363
PPO loss                     14.5397
value loss                   29.1149
吞吐                         73.17 world-step/s
PyTorch peak allocated       1.6183 GiB
PyTorch peak reserved        1.9355 GiB
```

训练过程中未出现 NaN/Inf 或 CUDA 异常。最终训练成功率明显提高，critic explained
variance 达到约 0.889；但后期 gradient norm 与 value loss 仍偏高，应结合独立评估的
行为副作用继续优化，而不是只看训练碰撞成功率。

训练指标和 TensorBoard：

```text
/home/user/whz/agent/outputs/mappo_rtx4090/metrics.jsonl
/home/user/whz/agent/outputs/mappo_rtx4090/tensorboard
```

## 12. Checkpoint

输出目录：

```text
/home/user/whz/agent/outputs/mappo_rtx4090/checkpoints
```

共保存 21 个 `.pt` 文件，包括 20 个周期 checkpoint 和 `final.pt`。最后几个为：

```text
step_004505600.pt
step_004751360.pt
step_005005312.pt
final.pt
```

最终 checkpoint：

```text
路径    /home/user/whz/agent/outputs/mappo_rtx4090/checkpoints/final.pt
大小    3,989,874 bytes
SHA-256 604145f22748fd589dc6b0fa494d01d60eb3da329d0f727a50cc691957d59b8f
```

已使用 `torch.load(..., map_location="cpu")` 成功读取，包含：

```text
config
model
optimizer
global_world_steps
update
python_rng_state
numpy_rng_state
torch_rng_state
cuda_rng_state
```

因此该文件可用于恢复模型、优化器、global step 和随机数状态。

## 13. 16 场景独立评估

评估摘要：

```text
/home/user/whz/agent/outputs/mappo_rtx4090/evaluation/eval_summary.json
```

结果：

| 指标 | 数值 |
|---|---:|
| worlds | 16 |
| active attackers | 16 |
| stage | direct_16 |
| ego collision rate | 0.9375 |
| mean collision time | 2.5933 s |
| minimum distance mean | 2.0759 m |
| minimum TTC mean | 0.5152 s |
| non-ego collision rate | 0.2585 |
| offroad rate | 0.2642 |

积极结果是策略在 scene-disjoint eval 中仍有很高的 ego 碰撞触发率和较低 TTC。主要问题
是 offroad 与 non-ego collision 同样较高，说明策略仍会通过不够合理的激进行为取得成功。

后续建议按优先级：

1. 提高 offroad、road collision 和 non-ego collision 的负奖励；
2. 分别绘制四个课程阶段的安全指标，确认副作用从哪个阶段开始上升；
3. 在 DIRECT 阶段降低动作饱和，增加 action delta / steering 正则；
4. 增加按原始 log 分组的 log-disjoint 划分；
5. 引入闭环 ego 规划器后再讨论对真实规划策略的攻击泛化。

## 14. 可视化验收

评估轨迹与媒体：

```text
/home/user/whz/agent/outputs/mappo_rtx4090/evaluation/eval_trace.csv
/home/user/whz/agent/outputs/mappo_rtx4090/evaluation/rollout.gif
/home/user/whz/agent/outputs/mappo_rtx4090/evaluation/rollout.mp4
/home/user/whz/agent/outputs/mappo_rtx4090/evaluation/final_frame.png
```

验收：

| 文件 | 尺寸/帧数 | 时长 | 大小 | 结果 |
|---|---|---:|---:|---|
| rollout.gif | 1600×900，100 帧 | 10 s | 6,464,310 bytes | PASS |
| rollout.mp4 | 200 帧 | 10.00 s | 5,808,501 bytes | PASS |
| final_frame.png | 1600×900 | — | 239,076 bytes | PASS |
| eval_trace.csv | — | — | 2,340,302 bytes | 生成成功 |

Pillow 能打开并验证 GIF/PNG，`imageio-ffmpeg` 检测 MP4 为 200 帧、10 秒。人工查看最终
帧确认真实 HD Map、道路边界、车道线、ego 和攻击车辆可辨认；多辆攻击车聚集到 ego
附近，与高碰撞率和较高副作用指标一致。

媒体 SHA-256：

```text
eval_summary.json  74cea1a56692a17b3ce56ddaad533234428be77c9e8a57c7fcdd35a21f0c1eca
rollout.gif        bcb6497c01dd8c20702353d8b555960cb06082b02c3c847de062c06137488405
rollout.mp4        71457b8428e0cefe12c92aec30d4d78ccffa52693d43f1d48ac9307bdf434cfb
```

## 15. 输出目录与磁盘占用

```text
CanonicalScenario            /data/user/agent_runs/nuplan_mappo_128/canonical
                              约 1.2 GiB

RuntimeScenario              /data/user/agent_runs/nuplan_mappo_128/runtime
                              约 110 MiB

训练与评估输出               /home/user/whz/agent/outputs/mappo_rtx4090
                              约 97 MiB

完整流水线日志               /data/user/agent_runs/nuplan_mappo_128/pipeline.log
                              906,014 bytes
```

Git 仓库只提交本报告、构建修复和一键编排脚本；数据、checkpoint、CSV、TensorBoard、
GIF 和 MP4 不提交，避免把大型二进制产物写入源码历史。

## 16. 恢复、评估与 TensorBoard

从最后周期 checkpoint 恢复训练：

```bash
cd /home/user/whz/agent
export PATH=/usr/local/cuda-12.4/bin:/home/user/anaconda3/envs/gpudrive/bin:/usr/bin:/bin
export CUDA_VISIBLE_DEVICES=0

./gpudrive_cuda/scripts/run_mappo_rtx4090.sh \
  outputs/mappo_rtx4090/checkpoints/step_005005312.pt
```

只重新评估：

```bash
cd /home/user/whz/agent
export PYTHONPATH="$PWD/build/gpudrive_cuda_rl/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"

/home/user/anaconda3/envs/gpudrive/bin/uv run --frozen --group train \
  python -m gpudrive_cuda.rl.evaluate \
  --config gpudrive_cuda/configs/mappo_rtx4090.json \
  --checkpoint outputs/mappo_rtx4090/checkpoints/final.pt \
  --output outputs/mappo_rtx4090/evaluation \
  --render
```

启动 TensorBoard：

```bash
cd /home/user/whz/agent
./gpudrive_cuda/scripts/run_tensorboard.sh
```

本地访问：

```bash
ssh -N -L 6006:127.0.0.1:6006 user@workstation
```

浏览器打开 `http://127.0.0.1:6006`。

## 17. 最终结论

本次运行完整打通了：

```text
真实 nuPlan SQLite + HD Map
  -> 128 CanonicalScenario
  -> 128 固定容量 RuntimeScenario
  -> scene-disjoint 64 train / 16 eval
  -> CUDA 12.4 Torch bridge
  -> RTX 4090 MAPPO 5,005,312 world-step
  -> checkpoint / TensorBoard / JSON / CSV / GIF / MP4 / PNG
```

数据转换、验证、划分、CUDA bridge 测试、训练、checkpoint 和媒体生成全部成功，流水线
退出码为 0。策略在 16 个独立 scene 上取得 93.75% ego 碰撞率，但 26.42% offroad 和
25.85% non-ego collision 表明当前行为合理性仍不足。后续工作的重点应从继续追高碰撞率，
转向降低越界和无关碰撞，并使用 log-disjoint 数据与闭环 ego 规划器验证泛化能力。
