# VADM — Variation-Aware SDR Mapping for Mixed-Precision FeFET-CIM

本模块独立于主项目（`../src/`、`../config/`），不依赖主项目任何模块，可单独运行。

---

## 1. 背景与动机

混合精度 FeFET-CIM 阵列中，每个 INT8 权重 $w_{i,j}$ 通过**有符号数字表示（SDR）**映射到 $K_B$ 个 1-bit PE 平面与 $K_Q$ 个 2-bit PE 平面上。由于同一权重存在多种等价的 SDR 分解方案，不同分解会导致各列的**模拟累加分布**不同，进而影响量化读出的误判概率（PE error）。

本模块的目标是：在给定精度配置 $(K_B, K_Q)$ 下，通过**逐列贪婪坐标下降**为每个权重选择最优 SDR 候选，**直接最大化 CIM 列级精度**：

$$\max_{\{d_{i,j}\}}\; \sum_c \log P_c$$

其中 $P_c$ 是列 $c$ 在全1输入下的解析正确概率，由 FeFET 器件物理统计直接计算（无需 Monte Carlo）。

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
├── plot_inno2.py               # 论文配图生成
├── README.md
│
├── src/                        # 库模块（由入口脚本通过 sys.path 导入）
│   ├── config_inno2.py         # 所有参数配置（唯一需要修改的文件）
│   ├── FeFET_model.py          # FeFET 物理器件变异模型（1F1R，支持 1-bit/2-bit）
│   │
│   │   ── Phase 1：校准 ──
│   ├── analytical_cal.py       # 快速解析校准（LUT 积分 + 高斯近似，推荐）
│   ├── col_mc_sim.py           # MC 列仿真（Phase 1.1，慢速验证模式）
│   ├── error_prob.py           # MC 路径：误判概率曲线 + N_th（Phase 1.2）
│   ├── kappa_calibration.py    # MC 路径：kappa 拟合（Phase 1.3）
│   │
│   │   ── Phase 2/3：分解与 Mapping ──
│   ├── decompose.py            # SDR 分解：LUT、conventional、twos_complement、min-neq
│   ├── column_stats.py         # 列统计：n^{1b}、n^{2b,eq}、J_c、S_c、目标函数
│   ├── baseline_mapping.py     # 基线 mapping（twos_complement_map、conventional_map、minneq_map）
│   ├── mapping_optimizer.py    # 代理目标优化器（保留，不在主对比链路）
│   ├── col_log_prob.py         # 解析列概率核心函数（p_err 表、log P_c 计算）
│   ├── accuracy_optimizer.py   # Proposed1：max Σ log P_c
│   ├── mac_accuracy_optimizer.py    # Proposed2：all-ones E2E 目标（保留）
│   ├── mac_accuracy_optimizer_p3.py # Proposed3：CRN 对齐 E2E 目标（保留）
│   │
│   │   ── Phase 4：精度评估 ──
│   └── cim_accuracy.py         # CIM bit-serial MAC 精度仿真（三路：TC/Conv/Proposed）
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

默认使用**快速解析校准**（`FAST_MODE=True`），Phase 1 在数秒内完成。整个流程依次执行 Phase 1–4，在一个随机 $64\times64$ 权重块上对比四种方法：

| 方法 | 模块 | 优化目标 |
|:----:|:----:|:--------:|
| **TC（二补码）** | `baseline_mapping.py` | 无优化（最高位为符号位，低位非负；最传统数字存储方式） |
| **Conventional（差分幅值码）** | `baseline_mapping.py` | 无优化（符号幅值分解，正负极板各自独立） |
| Min-neq SDR | `baseline_mapping.py` | 最小化等效活跃数 |
| **Proposed1** | `accuracy_optimizer.py` | $\max \sum_c \log P_c$（all-ones bit-plane objective） |

Phase 4 输出**三路精度对比**：TC / Conventional / Proposed1。

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

需先运行 `run_inno2.py` 生成 `Results/` 中的数据文件。

### 4.4 单独测试精度直接优化器

```bash
python src/accuracy_optimizer.py
```

在随机权重块上运行 5 轮，打印每轮 $\sum_c \log P_c$ 的变化（应单调递增）。

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

### 6.1 三种 Mapping 方案对比

| 方案 | 负权重存储方式 | B_minus 激活情况 | B_plus/Q_plus 激活情况 |
|:----:|:-------------:|:----------------:|:---------------------:|
| **TC（二补码）** | 最高位 $d^B_{K_B-1}=-1$（符号位），低位非负 | 仅 MSB 平面 | 所有低位平面均可激活 |
| **Conventional（差分幅值码）** | 全部数字取负 | 所有非零平面 | 无 |
| **Proposed1** | 优化后：尽量令所有平面全 1（对齐全局最优） | 见优化结果 | 见优化结果 |

**关键区别**：对于负权重 $w \in [-\lambda_{K_B-1},\, -1]$（即 $[-64, -1]$），TC 使低位正极板（B_plus/Q_plus）保持活跃，列活跃数更高，导致 CIM 量化误差更大；而差分幅值码只激活负极板，列活跃数更低。Proposed1 通过直接优化 $\sum_c \log P_c$ 进一步降低误差。

> **TC 的表示范围**：在默认配置（$K_B=3, K_Q=2$，$\lambda_{K_B-1}=64$）下，严格二补码仅能表示 $[-64, 127]$；对于 $w < -64$ 的权重，自动回退到符号幅值分解（两者在此范围等价）。

### 6.2 SDR 分解

每个 INT8 权重 $w \in [-127, 127]$ 被分解为：

$$w = \sum_{m=0}^{K_B-1} \lambda_B[m]\, d^B_m \;+\; \sum_{t=0}^{K_Q-1} \lambda_Q[t]\, d^Q_t$$

其中：
- $\lambda_B[m] = 2^{2K_Q+m}$，$d^B_m \in \{-1,0,+1\}$（1-bit PE 有符号数字）
- $\lambda_Q[t] = 4^t$，$d^Q_t \in \{-3,-2,-1,0,+1,+2,+3\}$（2-bit PE 有符号数字）

正/负平面（互不重叠）：
$$B^+_m = \max(d^B_m, 0),\quad B^-_m = \max(-d^B_m, 0)$$
$$Q^+_t = \max(d^Q_t, 0),\quad Q^-_t = \max(-d^Q_t, 0)$$

### 6.3 精度直接优化器（主算法）

#### 核心目标

在**全1输入**（最大负载场景）下，每列 $c$、每个 bit-plane $b$、每侧（$+/-$）的模拟累加和服从高斯分布：

$$S^\text{phys} \sim \mathcal{N}(\mu_S,\, \sigma^2_S)$$

其中 $\mu_S$、$\sigma^2_S$ 由该列的 cell 状态组成（通过 FeFET 单器件统计量求和）直接确定：

**1-bit 平面**，列 $c$ 有 $n_\text{on}$ 个 ON 态 cell：
$$\mu_S = n_\text{on}\cdot\mu_\text{on} + (M-n_\text{on})\cdot\mu_\text{off}, \quad \sigma^2_S = n_\text{on}\cdot\sigma^2_\text{on} + (M-n_\text{on})\cdot\sigma^2_\text{off}$$

**2-bit 平面**，列 $c$ 各状态数量为 $(n_0, n_1, n_2, n_3)$：
$$\mu_S = \sum_{s=0}^{3} n_s\cdot\mu_s, \quad \sigma^2_S = \sum_{s=0}^{3} n_s\cdot\sigma^2_s, \quad S_\text{ideal} = n_1 + 2n_2 + 3n_3$$

PE 量化误判概率（全1输入下，决策边界为整数 $\pm 0.5$）：

$$p_\text{err} = \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{S_\text{ideal}+0.5-\mu_S}{\sqrt{2}\,\sigma_S}\right) + \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\mu_S - S_\text{ideal}+0.5}{\sqrt{2}\,\sigma_S}\right)$$

**列级对数概率**（对所有 bit-plane 和正负两侧求和）：

$$\log P_c = \sum_{m=0}^{K_B-1}\left[\log(1-p^+_{m,c}) + \log(1-p^-_{m,c})\right] + \sum_{t=0}^{K_Q-1}\left[\log(1-p^+_{t,c}) + \log(1-p^-_{t,c})\right]$$

利用**列独立性**（各列电荷积分互不影响），全局优化目标分解为：

$$\max_{\{d_{i,j}\}}\; \sum_c \log P_c \;\;\Longleftrightarrow\;\; \max_c \log P_c \text{（逐列独立最大化）}$$

#### 状态追踪（增量更新）

维护四个计数数组：

| 数组 | 形状 | 含义 |
|:----:|:----:|:----:|
| `n_on_p[m, c]` | $[K_B, N]$ | 1-bit 平面 $m$ 列 $c$ 的 $B^+$ ON 态数量 |
| `n_on_m[m, c]` | $[K_B, N]$ | 1-bit 平面 $m$ 列 $c$ 的 $B^-$ ON 态数量 |
| `n_state_p[t, c, s]` | $[K_Q, N, 4]$ | 2-bit 平面 $t$ 列 $c$ 中 $Q^+$ 状态 $s$ 的数量 |
| `n_state_m[t, c, s]` | $[K_Q, N, 4]$ | 2-bit 平面 $t$ 列 $c$ 中 $Q^-$ 状态 $s$ 的数量 |

当元素 $(i,c)$ 的 SDR 系数从旧候选切换到新候选时，仅需 $O(K_B + K_Q)$ 的增量更新，无需重扫整列。

#### 算法流程

```
预计算: FeFET 器件统计量 (μ_s, σ²_s)  ← compute_device_stats()
预计算: 1-bit p_err 查找表 [M+1]       ← build_p_err_table_1bit()

初始化: mapping ← conventional 二进制分解
初始化: 计数数组 n_on_p/m, n_state_p/m  ← _init_count_arrays()
初始化: log_P[c] for all c

repeat:
    row_order = shuffle(0..M-1)
    improved = False

    for i in row_order:
        for c in range(N):
            for cand in lut[W[i,c]]:
                Δ计数 = O(1) 增量（仅 1-bit 加减、2-bit 直方图修改）
                try_log_P_c = col_log_prob(...)   ← O(K_B + K_Q)
                if try_log_P_c > log_P[c] + ε:
                    记录最优候选

            if 找到更优候选:
                提交更新（D_B, D_Q, 计数数组, log_P[c]）
                improved = True

until not improved
```

#### 与代理目标优化器的本质区别

| 对比项 | 代理目标（旧） | 精度直接（新） |
|:------:|:-------------:|:-------------:|
| 优化目标 | $\min \sum_c (J_c + \eta S_c)$ | $\max \sum_c \log P_c$ |
| 物理含义 | 启发式过载惩罚 + 稀疏性 | 解析量化正确概率 |
| 参数敏感性 | 需调 $\eta$、overload/sparsity mode | 只需 FeFET 噪声模型 |
| 理论保证 | 代理与真实目标相关性弱 | 单调优化，理论等价于 CIM 精度 |
| 系统偏差捕获 | 仅通过 $N_\text{th}$ 间接近似 | 直接计算包含系统偏差的完整分布 |

### 6.4 Phase 1 校准——确定器件统计量

**快速模式**（`FAST_MODE=True`，默认）：

基于 1F1R LUT 数值积分（每状态 400 点高斯权重）直接计算单器件电荷统计量 $(\mu_Q, \sigma^2_Q)$，再利用列求和为正态分布（CLT，$M=64$ 时精度极好）解析计算：

$$p_\text{err}(n) = \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\Delta\mu}{2\sqrt{2}\,\sigma_n}\right) + \frac{1}{2}\,\mathrm{erfc}\!\left(\frac{\Delta\mu}{2\sqrt{2}\,\sigma_{n+1}}\right)$$

$N_\text{th}$ 定义为满足 $p_\text{err}(n) \leq \varepsilon$ 的最大 $n$。$\kappa$ 直接由电荷方差比给出，全程无 Monte Carlo，数秒完成。

**MC 模式**（`FAST_MODE=False`，用于验证）：

| 对比项 | 快速模式 | MC 模式 |
|:------:|:--------:|:-------:|
| Phase 1 耗时 | **数秒** | 数小时（~47905 combo） |
| 内存占用 | **< 1 MB** | ~ 960 MB |
| 精度 | 高斯近似（$M=64$ 极好） | 精确尾概率 |

---

## 7. Phase 4 — CIM MAC 精度评估（量化整数模型）

**固定芯片模型**：$V_\text{th}$ 变异在权重矩阵级别采样一次，所有测试向量复用同一套器件特性。

**三步 PE 计算模型**：

1. **模拟积分**：$S^\text{phys}[j] = \sum_i x^\text{bit}_k[i]\cdot Q^\text{eff}_\text{cell}[i,j]$

2. **量化读出**（ADC / Sense-Amp）：$S^\text{quant}[j] = \operatorname{clip}(\operatorname{round}(S^\text{phys}[j]),\; 0,\; C)$
   — 若 $S^\text{quant}[j] \neq S^\text{ideal}[j]$，记为一次 **PE 量化错误**

3. **位权加权累积**：$y_\text{CIM}[j] \mathrel{+}= w_k \cdot \sum_m \lambda_B[m]\,(S^\text{quant}_{+,m} - S^\text{quant}_{-,m}) + \cdots$

**精度评估**（对 1000 个随机 INT8 向量统计，三方对比）：

| 指标 | 说明 |
|------|------|
| **MAC exact match rate** | $\operatorname{round}(y_\text{CIM}[j]) = y_\text{ref}[j]$ 的列比例 |
| 余弦相似度 | $\cos(y_\text{CIM},\, y_\text{ref})$ 均值/标准差 |
| 相对 L2 误差 | $\|y_\text{CIM} - y_\text{ref}\|_2 / \|y_\text{ref}\|_2$ 均值/标准差 |
| Per-plane PE 错误率 | 各 bit-plane 量化错误次数占比 |

Phase 4 输出**三路对比**：**TC（二补码）vs Conventional（差分幅值码）vs Proposed1**。

---

## 8. 结果文件（`Results/`）

| 文件 | 内容 | 生成来源 |
|------|------|----------|
| `calibration_params.npz` | $N^{(1b)}_\text{th}$，$N^{(2b)}_\text{th}$，$\kappa_2$，$\kappa_3$ | Phase 1 |
| `kappa_final.npz` | 同上 + $p_\text{err}$ 曲线数组 | Phase 1 |
| `error_prob_1bit.npz` | $n$，$p^{(1b)}_\text{err}(n)$ | Phase 1 |
| `error_prob_2bit.npz` | $n_\text{eq}$，$p^{(2b)}_\text{err}(n_\text{eq})$ | Phase 1 |
| `result_tc.npz` | TC（二补码）mapping 统计量 | Phase 2 |
| `result_conventional.npz` | Conventional（差分幅值码）mapping 统计量 | Phase 2 |
| `result_minneq.npz` | Min-neq SDR mapping 统计量 | Phase 2 |
| `result_proposed1.npz` | Proposed1 mapping 统计量 | Phase 3 |
| `cim_accuracy.npz` | 三路精度对比（TC/Conv/Proposed，exact match、余弦、L2、PE 错误率） | Phase 4 |

---

## 9. 依赖

```
numpy
scipy        # erfc, brentq（Phase 1 校准 & 精度直接优化器）
matplotlib   # 图表生成（plot_inno2.py）
torch        # FeFET 变异采样（Phase 4，CPU 即可）
```

`torch` 仅在 Phase 4 的 `cim_accuracy.py` 和 `FeFET_model.py` 中使用；若只运行 Phase 1–3，无需安装。
