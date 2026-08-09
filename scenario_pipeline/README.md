# nuPlan / Waymo 统一场景接口

`scenario_pipeline` 将两种原始数据转换为 `CanonicalScenario v1` JSON：

```text
nuPlan SQLite + map.gpkg ─┐
                          ├──> CanonicalScenario JSON
Waymo Scenario TFRecord ──┘
```

公共输出使用 10 Hz 时间轴和局部坐标：初始 ego 位于 `(0, 0)`，朝向
`+X`，车身左侧为 `+Y`。无效状态保存为 `valid=false`，对应数值为
`null`。

## nuPlan

在 nuPlan devkit v1.2.2 环境中运行：

```bash
python -m scenario_pipeline convert-nuplan \
  --input /data/nuplan/nuplan-v1.1/splits/mini \
  --maps-root /data/nuplan/maps \
  --output /data/canonical/nuplan
```

不传 `--maps-root` 时仍会转换轨迹，但 `quality.map_available=false`。

### 转换前后可视化

安装 `matplotlib` 后，可以把原始 nuPlan UTM 轨迹和转换后的 ego 局部坐标并排画出：

```bash
python -m scenario_pipeline visualize-nuplan \
  --input /data/nuplan/nuplan-v1.1/splits/mini/example.db \
  --maps-root /data/nuplan/maps \
  --output /data/canonical/nuplan/example_before_after.png
```

命令同时生成同名 `.md` 摘要。没有地图时仍能画轨迹；有地图且 devkit
成功加载时，右图会叠加 lane、road edge、crosswalk 和 stop line。

TerraZero 风格的动态俯视轨迹回放：

```bash
python -m scenario_pipeline animate-nuplan \
  --input /data/nuplan/nuplan-v1.1/splits/mini/example.db \
  --maps-root /data/nuplan/maps \
  --output /data/canonical/nuplan/example_trajectory.gif \
  --fps 10
```

蓝色矩形表示 ego，橙色矩形表示其他车辆，实线为最近一段历史轨迹，
淡色虚线为完整记录轨迹。`--stride 2` 可隔帧渲染以减小 GIF。

需要严格检查转换前后是否一致时，生成同步对照动画：

```bash
python -m scenario_pipeline compare-nuplan \
  --input /data/nuplan/nuplan-v1.1/splits/mini/example.db \
  --maps-root /data/nuplan/maps \
  --output /data/canonical/nuplan/example_comparison.gif
```

该命令输出并排对照 GIF，以及后缀为 `_before.gif` 和 `_after.gif` 的
两个独立 GIF。原始低频帧使用 sample-and-hold，Canonical 侧按 10 Hz
播放；标题中的 pairwise-distance max error 用于检查刚体变换是否保持
车辆之间的距离。

## Waymo

使用 Python 3.10，并安装：

```bash
pip install waymo-open-dataset-tf-2-12-0
```

然后运行：

```bash
python -m scenario_pipeline convert-waymo \
  --input /data/waymo/scenario/training \
  --output /data/canonical/waymo
```

输入必须是官方 Motion Dataset `Scenario` protobuf TFRecord，不是
`tf.Example` 格式。

## 验证

```bash
python -m scenario_pipeline validate --input /data/canonical
python -m unittest discover -s scenario_pipeline/tests -v
```

批处理会继续处理其他文件，并在结尾输出成功、警告和失败数量。

## 编译为 RL Runtime

Canonical JSON 不直接进入 CUDA。先编译为固定容量、连续内存张量：

```bash
python -m scenario_pipeline compile-rl \
  --input /data/canonical/nuplan \
  --output /data/rl_runtime/nuplan

python -m scenario_pipeline validate-rl \
  --input /data/rl_runtime/nuplan
```

输出格式见 [RUNTIME_FORMAT.md](RUNTIME_FORMAT.md)。真实未来轨迹单独保存为
`reference_future`，并明确标记为不可进入 policy observation。

## 主要字段

```text
schema_version
source
timing
coordinate_frame
agents
map_features
traffic_lights
route
tags
quality
```

nuPlan 提供 route；Waymo 没有等价的任务路线，因此明确输出
`route.available=false`，不会用历史轨迹伪造路线。
