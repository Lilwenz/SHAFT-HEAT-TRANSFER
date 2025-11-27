# 三维阶梯轴 PINN 热传导预测

本目录针对**三段式阶梯轴**的瞬态热传导问题，基于 Abaqus 有限元仿真数据，构建物理信息神经网络（Physics-Informed Neural Network, PINN），在仅使用少量温度传感器数据的情况下，实现对全场温度场的高精度预测与可视化。

## 1. 项目结构

当前目录下关键文件与子目录：

- `extract_temperature.py`  
  使用 Abaqus Python 接口，从 `../shaft/Job-4.odb` 中导出节点温度随时间的历史数据到 CSV。
- `job4_temperatures__SENSORS-N.csv`  
  仅包含少量传感器节点的温度数据，用于 PINN 的“数据损失”部分。
- `job4_temperatures__ALL-N.csv`  
  包含全场所有节点在整个时间区间的温度数据，用于 PDE 物理约束和最终验证。
- `shaft_heat_pinn.ipynb`  
  主要的实验 Notebook，完成数据加载、几何分区、PINN 模型定义、训练与评估、可视化等。
- `weights/`  
  训练生成的模型权重归档（带时间戳的 `shaft_heat_pinn_*.pth`、Lightning `*.ckpt` 等）。
- `plot/`  
  按时间戳划分的评估结果文件夹，包含误差统计表、抽样对比 CSV、2D 分析曲线、3D 交互式可视化等。
- `requirements.txt`  
  运行 Notebook 所需的 Python 依赖列表。

该目录可以直接作为一个独立子仓库上传到 GitHub。

## 2. 物理问题与数据来源

### 2.1 几何与边界条件

- 几何：三段阶梯轴  
  - 段 1：直径 50 mm，长度 150 mm  
  - 段 2：直径 100 mm，长度 700 mm  
  - 段 3：直径 50 mm，长度 150 mm  
  - 轴向坐标区间约为 \(z \in [0, 1000]\) mm。
- 材料：45 号钢（Steel 45），采用统一单位体系：
  - 长度：mm
  - 时间：s
  - 温度：℃  
  - 密度：\(\rho = 7.9\times 10^{-9}\) tonne/mm³  
  - 比热容：\(c_p = 4.5\times 10^{8}\) mJ/(tonne·℃)  
  - 导热系数：\(k = 48\) mW/(mm·℃)
- 边界条件：
  - **端面 (Ends, z = 0 / 1000)**：给定热流 \(q_n\) 与对流换热系数 \(h\)，满足
    \[ k\frac{\partial T}{\partial n} = q_n - h(T - T_\infty) \]
  - **侧面 (Sides)**：圆柱侧面 \(r=25\) mm 或 \(r=50\) mm，仅对流散热
    \[ k\frac{\partial T}{\partial n} = -h(T - T_\infty) \]
  - **台阶面 (Steps, z=150 / 850)**：环形台阶面，仅对流散热
    \[ k\frac{\partial T}{\partial n} = -h(T - T_\infty) \]

### 2.2 控制方程（PDE）

三维瞬态导热方程：

\[
\rho c_p \frac{\partial T}{\partial t}
= k \nabla^2 T
= k \left( \frac{\partial^2 T}{\partial x^2}
+ \frac{\partial^2 T}{\partial y^2}
+ \frac{\partial^2 T}{\partial z^2} \right)
\]

PINN 在内部区域通过最小化 PDE 残差来学习物理规律，在边界上通过法向导数与对流换热条件构造边界残差。

### 2.3 数据文件说明

- `job4_temperatures__SENSORS-N.csv`  
  - 每一行包含：时间步索引、时间、节点集合名、实例名、节点编号、坐标 \((x,y,z)\)、温度 `NT11`。  
  - 仅覆盖少数传感器节点，用作“观测数据”。
- `job4_temperatures__ALL-N.csv`  
  - 包含全场所有节点，覆盖全部时间步。  
  - 用于构造 PDE 配点、边界点（端面、台阶面、侧面）以及最终误差评估。

## 3. 软件依赖与环境

本目录提供了 `requirements.txt`，可一键安装主要依赖：

```bash
pip install -r requirements.txt
```

其中主要包括：

- `torch`
- `lightning`
- `pinnstorch`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `ipywidgets`
- `nbformat>=4.2.0`

如果你希望自己从 Abaqus `.odb` 导出温度数据，还需要：

- 已安装 Abaqus/CAE 或 Abaqus/Standard
- 能够使用 Abaqus 自带的 Python（带 `odbAccess` 和 `abaqusConstants`）

> 建议在 Conda 或虚拟环境中安装上述依赖。

## 4. 快速开始

### 4.1 克隆与安装

```bash
git clone https://github.com/<your-name>/<your-repo>.git
cd <your-repo>/shaft-heat-transfer
pip install -r requirements.txt
```

然后在 VS Code / Jupyter 中打开 `shaft_heat_pinn.ipynb`，按顺序运行各个单元即可复现训练与评估流程。

### 4.2 从 Abaqus 导出温度数据（可选）

如果已经有仓库中现成的 `job4_temperatures__SENSORS-N.csv` 和 `job4_temperatures__ALL-N.csv`，可以直接跳过本步。  
如需从 ODB 重新导出数据：

```bash
abaqus python extract_temperature.py ^
  --odb ..\shaft\Job-4.odb ^
  --step HeatStep ^
  --node-sets SENSORS-N ALL-N ^
  --split-node-sets ^
  --output ..\shaft-heat-transfer\job4_temperatures.csv
```

脚本会根据 `--split-node-sets` 生成：

- `job4_temperatures__SENSORS-N.csv`
- `job4_temperatures__ALL-N.csv`

## 5. PINN 训练流程（在 Notebook 中）

主要流程均在 `shaft_heat_pinn.ipynb` 中实现，核心步骤如下：

1. **数据加载与几何分组**  
   - 读取 `SENSORS-N` 与 `ALL-N` 两个 CSV。  
   - 计算节点半径 \(r = \sqrt{x^2 + y^2}\)。  
   - 根据 \((r,z)\) 将全场节点划分为：端面（Ends）、台阶面（Steps）、侧面（Sides）、全场域（Domain）。
2. **数据结构封装**  
   - 将 `pandas.DataFrame` 转换为 `PointCloudData` / `PointCloud`（`pinnstorch.data`），支持 \(N\times T\) 的时空网格结构。
3. **物理参数与网络结构**  
   - 定义 \(\rho, c_p, k, q_n, h, T_\infty\) 等物理常数。  
   - 使用全连接网络 `FCN`，输入 \((x,y,z,t)\)，输出温度 `T`，并通过 Sigmoid+线性缩放限制在物理合理范围。
4. **PDE 与边界条件**  
   - 在 `heat_pde_fn` 中使用自动微分计算一二阶导数，构造：
     - `f_T`：内部 PDE 残差
     - `f_bc_ends`：端面边界残差
     - `f_bc_steps`：台阶面边界残差
     - `f_bc_sides`：侧面边界残差
   - 对困难区域（如台阶面）提高损失权重，加强训练关注度。
5. **采样器与 DataModule**  
   - 使用 `MeshSampler` 与 `InitialCondition` 构建多种损失项：
     - 传感器数据（Data Loss）
     - 初始条件（IC Loss）
     - PDE 残差（PDE Loss）
     - 端面/台阶面/侧面边界（BC Loss）
   - 使用 `PINNDataModule` 管理训练/验证/预测数据。
6. **训练配置与记录**  
   - 自定义 `PINNModuleWithLossLogging`，在每个 epoch 记录各个损失分量。  
   - 使用 Lightning `Trainer`，配置 `ModelCheckpoint` 自动保存最佳模型和最后模型。  
   - 使用 `LossHistoryCallback` 记录并绘制 loss 曲线，保存为 PNG 与 CSV。

训练完成后，会在 `weights/` 下生成带时间戳的模型权重，并在 `plot/` 下生成对应的评估结果文件夹（包含超参数、误差统计表、抽样对比数据、2D 曲线、3D 可视化等）。

## 6. 结果评估与可视化

Notebook 中包含完整的评估与可视化代码，典型输出包括：

- 各区域误差统计表 `metrics_summary.csv`
- 关键区域抽样对比 `sample_*.csv`
- 2D 分析图 `2d_analysis_curves.png`（轴向/径向分布 & 瞬态响应）
- 3D 交互式图 `3d_true_temp.html`、`3d_pred_temp.html`、`3d_error.html`

这些结果均保存在 `plot/eval_YYYYMMDD_HHMMSS/` 目录下。

## 7. 引用与致谢

本项目基于 PINN 框架 **pinnstorch** 进行开发与实验，如在论文或报告中使用本项目代码或思路，请在合适位置对 **pinnstorch** 进行引用和致谢。例如（根据你实际使用的版本和论文格式自行调整）：

> 使用了物理信息神经网络库 *pinnstorch*（https://github.com/XXXXXX/pinns-torch）进行三维阶梯轴瞬态热传导问题的建模与训练。

同时感谢 **pinnstorch** 作者及相关开源社区对 PINN 方法的实现与推广，为本工作的实现提供了重要基础。