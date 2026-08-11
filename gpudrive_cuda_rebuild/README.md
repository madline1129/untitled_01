# CUDA Simulator Rebuild

这是独立于正式 `gpudrive_cuda/` 的教学重建目录。

> 当前状态：WIP，不可独立构建。`runtime_scene.*` 已完成数据加载，
> `drive_sim.cuh` 已开始升级完整接口，但 `drive_sim.cu` 和 `main.cu` 仍停留在
> Stage 0，因此接口暂时不匹配。正式运行和动态自行车模型请使用
> `gpudrive_cuda/`。

该目录作为逐步学习快照提交，后续继续教学时再从当前断点恢复，不影响已经跑通
真实 nuPlan 的正式 simulator。

## Stage 0 包含什么

```text
固定数量的 world 和 Agent
一维 GPU 内存布局 [world, agent]
全部 reset 和选择性 world reset
二维加速度动作 [ax, ay]
点质量运动学 [x, y, vx, vy]
统一目标点
距离 reward 和 goal/timeout done
self observation
所有 partner observation
```

当前故意不包含：

```text
RuntimeScenario 和真实数据
yaw、steering 和自行车模型
自动控制器与外部控制 mask
车辆/道路碰撞
地图和交通灯
固定 16/64 observation
CSV、JSON 和渲染
```

## 构建状态

当前不要运行 `scripts/test_base.sh`。完成 `drive_sim.cu` 与新头文件的接口迁移后，
再恢复 CMake 构建和 smoke test。

## 当前执行顺序

```text
DriveSim 构造
  -> cudaMalloc
  -> reset_kernel
  -> observation kernels
  -> set_actions
  -> step_kernel
  -> observation kernels
  -> 复制回 CPU 打印
  -> 析构 cudaFree
```

Stage 0 使用一个全局 `step_count`。选择性 reset 会恢复指定 world 的状态，但不会
单独重置它的 episode 时钟。这个限制会在后续“独立 world 生命周期”阶段解决。
