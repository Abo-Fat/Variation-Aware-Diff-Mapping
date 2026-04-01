# VADM — Variation-Aware SDR Mapping for Mixed-Precision FeFET-CIM

本模块独立于主项目（`../src/`、`../config/`），不依赖主项目任何模块，可单独运行。

---

## 1. 背景与动机

混合精度 FeFET-CIM 阵列中，每个 INT8 权重 $w_{i,j}$ 通过**有符号数字表示（SDR）**映射到 $K_B$ 个 1-bit PE 平面与 $K_Q$ 个 2-bit PE 平面上。由于同一权重存在多种等价的 SDR 分解方案，不同分解会导致各列的**等效开启器件数**（active count）分布不同，进而影响列读取时的误判概率（overload error）。

本模块的目标是：在给定精度配置 $(K_B, K_Q)$ 下，通过**逐列贪婪坐标下降**（Algorithm 1）为每个权重选择最优 SDR 候选，最小化加权列过载风险与稀疏性惩罚之和：

$$\min_{\{d_{i,j}\}}\; \sum_c J_c + \eta \sum_c S_c$$

其中 $J_c$ 是列 $c$ 的过载风险（位权加权 hinge 惩罚），$S_c$ 是位权加权等效开启数（稀疏性项）。

---

## 2. 精度配置

约束：$K_B + 2K_Q = 7$，即 7 个 PE slot 在 1-bit 和 2-bit 间分配。

| 配置 $(K_B,\, K_Q)$ | 1-bit PE 数 | 2-bit PE 数 | 备注 |
|:-------------------:|:-----------:|:-----------:|:----:|
| $(7,\; 0)$ | 7 | 0 | 全 1-bit |
| $(5,\; 1)$ | 5 | 1 | — |
| $(3,\; 2)$ | 3 | 2 | **默认** |
| $(1,\; 3)$ | 1 | 3 | 全 2-bit |

默认配置：`KB=3, KQ=2`，在 [src/config_inno2.py](src/config_inno2.py) 中修改。

---

## 3. 文件结构

```
VADM/
├── run_inno2.py                # 全流程主入口（Phase 1–4）
├── plot_inno2.py               # 论文配图生成（Figure 1, Figure 2）
├── visualize_weight_bitplanes.py  # 权重 bit-plane 可视化工具
├── README.md
│
├── src/                        # 库模块（由入口脚本通过 sys.path 导入）
│   ├── config_inno2.py         # 所有参数配置（唯一需要修改的文件）
│   ├── FeFET_model.py          # FeFET 物理器件变异模型（1F1R，支持 1-bit/2-bit/mixed）
│   │
│   │   ── Phase 1：校准 ──
│   ├── analytical_cal.py       # 快速解析校准（LUT 积分 + 高斯近似，推荐）
│   ├── col_mc_sim.py           # MC 列仿真（Phase 1.1，慢速验证模式）
│   ├── error_prob.py           # MC 路径：误判概率曲线 + N_th（Phase 1.2）
│   ├── kappa_calibration.py    # MC 路径：kappa 拟合（Phase 1.3）
│   │
│   │   ── Phase 2/3：分解与 Mapping ──
│   ├── decompose.py            # SDR 分解：LUT 构建、conventional、min-neq
│   ├── column_stats.py         # 列统计：n^{1b}、n^{2b,eq}、J_c、S_c、目标函数
│   ├── baseline_mapping.py     # 基线 mapping（conventional_map、minneq_map）
│   ├── mapping_optimizer.py    # 提出的逐列 SDR 优化器（Algorithm 1）
│   │
│   │   ── Phase 4：精度评估 ──
│   └── cim_accuracy.py         # CIM bit-serial MAC 精度仿真（余弦相似度、相对 L2 误差）
│
├── Results/                    # 中间数值结果（.npz，自动生成）
└── Figures/                    # 图片输出（plot_inno2.py 生成）
```

---

## 4. 快速开始

### 4.1 运行完整流程

```bash
cd VADM
python run_inno2.py
```

默认使用**快速解析校准**（`FAST_MODE=True`），Phase 1 在数秒内完成。整个流程依次执行 Phase 1–4，在一个随机 $64\times64$ 权重块上比较 Conventional、Min-neq SDR 与 Proposed mapping 的列过载风险，并评估 CIM MAC 精度。

### 4.2 命令行选项

```bash
# 仅执行 Phase 1 校准，不运行 mapping
python run_inno2.py --phase1-only

# MC 模式下若仿真数据已存在则跳过
python run_inno2.py --skip-mc
```

### 4.3 生成论文配图

```bash
python plot_inno2.py
```

需先运行 `run_inno2.py` 生成 `Results/` 中的数据文件。输出：

- `Results/fig1_calibration.pdf/.png` — 器件电荷分布 & 误判概率曲线
- `Results/fig2_mapping.pdf/.png` — PE 占用率 & 列风险 $J_c$ 对比

### 4.4 可视化 bit-plane 分配

```bash
python visualize_weight_bitplanes.py                        # 默认 conventional
python visualize_weight_bitplanes.py --method minneq
python visualize_weight_bitplanes.py --method proposed --seed 2026
python visualize_weight_bitplanes.py --weight-npy path/to/W.npy
```

输出保存至 `Results/bitplane_viz_<method>/`。

### 4.5 单独运行各模块

```bash
python src/analytical_cal.py        # Phase 1 解析校准（自测）
python src/decompose.py             # SDR 分解自测
python src/mapping_optimizer.py     # 在随机权重块上对比三种 mapping
```

---

## 5. 配置参数（`src/config_inno2.py`）

### 精度与阵列参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `KB` | `3` | 每侧 1-bit PE 数量（MSB 侧） |
| `KQ` | `2` | 每侧 2-bit PE 数量（LSB 侧） |
| `W_MAX` | `127` | 量化权重范围 $[-127, 127]$ |
| `COLUMN_SIZE` | `64` | CIM 阵列列长 $M$ |

### 校准参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FAST_MODE` | `True` | `True`=解析校准；`False`=MC 校准 |
| `EPSILON` | `0.01` | 可靠性阈值 $\varepsilon$：$p_\text{err} \leq \varepsilon$ 定义 $N_\text{th}$ |
| `VTH_SIGMA_SCALE` | `1.0` | $V_\text{th}$ 变异幅度缩放（1.0 = 器件标称值） |
| `N_MC` | `5000` | MC 模式每配置仿真次数（仅 `FAST_MODE=False`） |
| `N_KAPPA_SAMPLES` | `500` | MC kappa 拟合时随机采样的 $(n_1,n_2,n_3)$ 组合数 |

### 优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `GAMMA` | `0.1` | （保留，原 worst-column 权衡系数） |
| `ETA` | `0.1` | 稀疏性惩罚系数 $\eta$（目标函数中 $\eta\sum_c S_c$ 项） |

### FeFET 器件参数（通常无需修改）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `FEFET_VTH_STATES` | `[-0.96, -0.53, -0.023, 0.52]` V | 4 个状态的标称阈值电压 |
| `FEFET_SIGMA_VTH` | `[54.6, 50.5, 61.9, 61.4]` mV | 各状态 $V_\text{th}$ 标准差 |
| `FEFET_SIGMA_R_REL` | `0.01` | 限流电阻 $R$ 的相对标准差（1%） |
| `FEFET_VD` | `0.1` V | 漏端电压 |
| `FEFET_R_LIMIT` | `1 MΩ` | 限流电阻 |
| `FEFET_T_STEP` | `50 ns` | 每步读取时间 |

---

## 6. 算法原理

### 6.1 SDR 分解

每个 INT8 权重 $w \in [-127, 127]$ 被分解为：

$$w = \sum_{m=0}^{K_B-1} \lambda_B[m]\, d^B_m \;+\; \sum_{t=0}^{K_Q-1} \lambda_Q[t]\, d^Q_t$$

其中：
- $\lambda_B[m] = 2^{2K_Q+m}$，$d^B_m \in \{-1,0,+1\}$（1-bit PE 有符号数字）
- $\lambda_Q[t] = 4^t$，$d^Q_t \in \{-3,-2,-1,0,+1,+2,+3\}$（2-bit PE 有符号数字）

正/负平面（互不重叠）：
$$B^+_m = \max(d^B_m, 0),\quad B^-_m = \max(-d^B_m, 0)$$
$$Q^+_t = \max(d^Q_t, 0),\quad Q^-_t = \max(-d^Q_t, 0)$$

对于给定的 $w$，满足上述分解的所有合法候选通过 SDR LUT（`build_sdr_lut`）预枚举。

### 6.2 列统计量

**1-bit PE 列活跃数**（正负平面合并）：

$$n^{(1b)}_{m,c} = \sum_i \left|d^B_m[i,c]\right|$$

**2-bit PE 等效活跃数**（$\varphi$ 函数将高 cell value 折算为等效噪声贡献）：

$$n^{(2b,\text{eq})}_{t,c} = \sum_i \varphi\!\left(\left|d^Q_t[i,c]\right|\right), \quad \varphi(v) = \begin{cases} 0 & v=0 \\ 1 & v=1 \\ \kappa_2 & v=2 \\ \kappa_3 & v=3 \end{cases}$$

其中 $\kappa_2,\kappa_3$ 由 Phase 1 校准确定（$\kappa_s = \sigma^2_{\text{cv}=s} / \sigma^2_{\text{cv}=1}$，即各 cell value 的电荷方差比）。

### 6.3 列风险与目标函数

**列过载风险**（$\alpha_m = \lambda_B[m]^2$，$\beta_t = \lambda_Q[t]^2$，高位平面被位权放大惩罚）：

$$J_c = \sum_m \alpha_m \left[n^{(1b)}_{m,c} - N^{(1b)}_\text{th}\right]_+ + \sum_t \beta_t \left[n^{(2b,\text{eq})}_{t,c} - N^{(2b)}_\text{th}\right]_+$$

**列稀疏性项**：

$$S_c = \sum_m \alpha_m\, n^{(1b)}_{m,c} + \sum_t \beta_t\, n^{(2b,\text{eq})}_{t,c}$$

**全局优化目标**：

$$\min_{\{d_{i,j}\}}\; \sum_c J_c + \eta \sum_c S_c$$

### 6.4 Phase 1 校准——确定 $N_\text{th}$ 与 $\kappa$

**快速模式**（`FAST_MODE=True`，默认）：

基于 1F1R LUT 数值积分（每状态 400 点 Gauss 权重）直接计算单器件电荷统计量 $(\mu_Q, \sigma^2_Q)$，再利用列求和为正态分布（CLT，$M=64$ 时精度极好）解析计算误判概率：

$$p_\text{err}(n) = \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\Delta\mu}{2\sqrt{2}\,\sigma_n}\right) + \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\Delta\mu}{2\sqrt{2}\,\sigma_{n+1}}\right)$$

$N_\text{th}$ 定义为满足 $p_\text{err}(n) \leq \varepsilon$ 的最大 $n$。$\kappa$ 直接由电荷方差比给出，无需任何 Monte Carlo 采样，全程数秒。

**MC 模式**（`FAST_MODE=False`，用于验证）：

逐步执行 Phase 1.1（列 MC 仿真）→ 1.2（误判概率曲线）→ 1.3（kappa 拟合），适合精确尾概率估计与交叉验证。

| 对比项 | 快速模式 | MC 模式 |
|:------:|:--------:|:-------:|
| Phase 1 耗时 | **数秒** | 数小时（~47905 combo） |
| 内存占用 | **< 1 MB** | ~ 960 MB |
| 精度 | 高斯近似（$M=64$ 时极好） | 精确尾概率 |
| 多进程 | 不需要 | 最多 8 核 |

### 6.5 Algorithm 1 — 逐列贪婪坐标下降

1. 以 **Min-neq SDR** 初始化：对每个元素独立选择等效活跃数最小的候选。
2. 计算初始列负载 $n^{(1b)}$、$n^{(2b,\text{eq})}$ 及目标函数值。
3. 重复至收敛（`max_iter` 轮）：
   - 随机打乱行序 $i$；
   - 对每个 $(i,\, c)$，枚举 LUT 中 $W[i,c]$ 的所有合法 SDR 候选；
   - 对每个候选，计算列 $c$ 的新目标值 $J_c + \eta S_c$（仅列 $c$ 受影响，$O(1)$ 增量更新）；
   - 若存在严格更优候选则提交更新，否则保留当前分配。
4. 目标函数不再下降时提前终止。

---

## 7. Phase 4 — CIM MAC 精度评估

**固定芯片模型**：$V_\text{th}$ 变异在权重矩阵级别采样一次，所有测试向量复用同一套器件特性，模拟真实芯片行为。

**Bit-serial 输入**：INT8 输入向量 $x$ 按两补码逐位分解（bit 0–6 权重为 $2^k$，bit 7 权重为 $-128$）。每个 bit 周期对 1-bit/2-bit PE 平面分别执行列求和并截断（clip $\pm M$ / $\pm 3M$），加权累积得到 CIM 输出 $y_\text{CIM}$。

**精度指标**（对 $N_\text{vec}=100$ 个随机 INT8 向量取均值和标准差）：
- 余弦相似度 $\cos(y_\text{CIM},\, y_\text{ref})$
- 相对 L2 误差 $\|y_\text{CIM} - y_\text{ref}\|_2 / \|y_\text{ref}\|_2$

---

## 8. 结果文件（`Results/`）

| 文件 | 内容 | 生成来源 |
|------|------|----------|
| `calibration_params.npz` | $N^{(1b)}_\text{th}$，$N^{(2b)}_\text{th}$，$\kappa_2$，$\kappa_3$ | Phase 1（两种模式均生成） |
| `kappa_final.npz` | 同上 + $p_\text{err}$ 曲线数组 | Phase 1 |
| `error_prob_1bit.npz` | $n$，$p^{(1b)}_\text{err}(n)$ | Phase 1 |
| `error_prob_2bit.npz` | $n_\text{eq}$，$p^{(2b)}_\text{err}(n_\text{eq})$ | Phase 1 |
| `result_conventional.npz` | Conventional mapping 的 $J_c$、$S_c$ 等统计量 | Phase 2 |
| `result_minneq.npz` | Min-neq SDR mapping 统计量 | Phase 2 |
| `result_proposed.npz` | Proposed mapping 统计量 | Phase 3 |
| `cim_accuracy.npz` | 余弦相似度与相对 L2 误差（baseline vs proposed） | Phase 4 |
| `col_mc_1bit.npz` | 1-bit MC 原始列分布 $Y[n_\text{on}, N_\text{mc}]$ | Phase 1.1（仅 MC 模式） |
| `col_mc_2bit.npz` | 2-bit MC 原始列分布 $Y[K_\text{combo}, N_\text{mc}]$ | Phase 1.1（仅 MC 模式） |

---

## 9. 依赖

```
numpy
scipy        # erfc, brentq（Phase 1 校准）
matplotlib   # 图表生成（plot_inno2.py）
torch        # FeFET 变异采样（Phase 4，CPU 即可）
```

`torch` 仅在 Phase 4 的 `cim_accuracy.py` 和 `FeFET_model.py` 中使用；若只运行 Phase 1–3，无需安装。
