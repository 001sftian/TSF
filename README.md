# 稀疏观测高阶怪波发现与大区域定位模型

本项目给出一个可落地扩展的研究原型：用小区域内稀疏的一阶怪波/波面观测作为上下文输入，训练一个 Transformer 模型去预测二、三、四、五阶怪波类别，并在更大空间-时间区域内输出定位热力图与中心坐标。

> 重要说明：仅靠真实的一阶怪波稀疏数据，通常无法可靠外推所有二到五阶怪波形态。实际科研或工程部署时，应混合使用高阶解析解、NLS/CFD 数值仿真、波浪水槽实验和现场传感器数据，并加入物理约束损失。本仓库先提供数据接口、网络结构和训练流程，便于把真实数据替换进来。

## 方法概览

1. **稀疏小区域上下文编码**：`context_coords` 和 `context_values` 表示少量传感器/采样点坐标及波幅。
2. **Transformer Encoder**：把稀疏坐标的 Fourier 特征与波幅拼成 token，学习局部观测中的非线性波动特征。
3. **任意大区域 Query Decoder**：把大区域网格坐标作为 query token，输出每个点属于怪波中心的 heatmap logit。
4. **阶数分类头**：对 encoder memory 做池化，预测一到五阶怪波类别。
5. **中心定位**：对 heatmap 做 soft-argmax，得到连续坐标中心。

## 文件结构

```text
src/rogue_locator/data.py      # 合成数据与批次 schema
src/rogue_locator/model.py     # Fourier 坐标编码 + SparseRogueWaveTransformer
src/rogue_locator/train.py     # 训练 CLI
src/rogue_locator/infer.py     # 推理 CLI 示例
tests/test_data_model.py       # 数据与模型形状测试
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

训练一个小型 baseline：

```bash
rogue-train --steps 500 --batch-size 8 --grid-size 48 --context-points 256 --output checkpoints/rogue_transformer.pt
```

用训练好的 checkpoint 在更大网格上推理：

```bash
rogue-infer --checkpoint checkpoints/rogue_transformer.pt --grid-size 96
```

## 换成真实/高保真数据的建议

`RogueWaveBatch` 是核心接口。只要真实数据或仿真数据能组织成下面字段，就可以直接复用模型和训练脚本：

- `context_coords`: `[B, P, 2]`，小区域稀疏观测坐标，例如 `(x, t)` 或 `(x, y)`。
- `context_values`: `[B, P, 1]`，对应波幅、包络强度或归一化自由液面高度。
- `grid_coords`: `[B, Q, 2]`，待搜索大区域网格坐标。
- `heatmap`: `[B, Q, 1]`，真实中心附近为 1 的定位标签。
- `order_labels`: `[B]`，一到五阶标签在代码中编码为 `0..4`。
- `centers`: `[B, 2]`，连续中心坐标监督。

进一步提高精度可加入：

- **物理约束**：加入 focusing NLS 残差、守恒量、边界条件损失。
- **多尺度 query**：先粗定位，再对候选区域高分辨率 refine。
- **不确定性估计**：Monte-Carlo dropout 或 ensemble，避免在一阶稀疏观测不足时给出过度自信定位。
- **真实域适配**：用仿真预训练，再用实验/现场数据微调。
- **类别均衡采样**：保证二、三、四、五阶样本足够，不让模型退化到只识别一阶。

## 当前合成数据的限制

`data.py` 中的波形是 Peregrine-like 的轻量代理模型，不是严格的高阶 NLS 解析解。它适合验证代码链路、训练接口和网络维度；若要发表论文或工程应用，应替换为经过验证的物理仿真/实验数据。
