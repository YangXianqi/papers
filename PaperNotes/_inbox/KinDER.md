---
title: "KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning"
method_name: "KinDER"
authors:
  - Yixuan Huang
  - Bowen Li
  - Vaibhav Saxena
  - Yichao Liang
  - Utkarsh Aashu Mishra
  - Liang Ji
  - Lihan Zha
  - Jimmy Wu
  - Nishanth Kumar
  - Sebastian Scherer
  - Danfei Xu
  - Tom Silver
year: 2026
venue: "arXiv"
tags: [robot-learning, physical-reasoning, robot-planning, benchmark, sim-to-real, manipulation]
zotero_collection: "_inbox"
image_source: online
arxiv: "http://arxiv.org/abs/2604.25788v1"
arxiv_pdf: "https://arxiv.org/pdf/2604.25788v1"
arxiv_html: "https://arxiv.org/html/2604.25788v1"
project_page: "https://prpl-group.com/kinder-site/"
code:
  - "https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden"
  - "https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines"
created: 2026-04-29
---

# 论文笔记：KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning

## 元信息

| 项目 | 内容 |
|------|------|
| 方法名 | KinDER, Kinematic and Dynamic Embodied Reasoning |
| 机构 | Princeton University, Carnegie Mellon University, Georgia Tech, University of Cambridge, NVIDIA, MIT |
| 日期 | 2026-04-28 |
| 领域 | [[Robot Learning]], [[Task and Motion Planning]], [[Physical Reasoning]], [[Sim-to-Real]] |
| 项目主页 | [KinDER](https://prpl-group.com/kinder-site/) |
| 代码 | [kindergarden](https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden), [kinder-baselines](https://github.com/Princeton-Robot-Planning-and-Learning/kinder-baselines) |
| 论文 | [arXiv](http://arxiv.org/abs/2604.25788v1), [HTML](https://arxiv.org/html/2604.25788v1), [PDF](https://arxiv.org/pdf/2604.25788v1) |

---

## 一句话总结

> KinDER 用 25 个程序生成环境系统测量机器人在运动学和动力学约束下的物理推理能力。

---

## 核心贡献

1. **KinDERGarden 环境集**: 提供 25 个可程序生成的 2D/3D 环境，覆盖基本空间关系、非抓取多物体操作、工具使用、组合几何约束、动力学约束五类 [[Embodied Physical Reasoning|具身物理推理]] 挑战。
2. **KinDERGym 软件接口**: 以 [[Gymnasium]] 兼容库封装环境、对象中心状态、参数化技能、概念接口、遥操作和演示数据，让规划、模仿学习、强化学习方法能在同一 API 下比较。
3. **KinDERBench 评测协议**: 实现 13 个代表性 baseline，横跨 [[Task and Motion Planning|任务与运动规划]]、[[Model Predictive Control|MPC]]、[[Reinforcement Learning|强化学习]]、[[Diffusion Policy|扩散策略]]、[[Vision-Language-Action Model|VLA]]、LLM/VLM 规划，并报告成功率、累计奖励、推理时间和工程成本。
4. **真实机器人映射验证**: 用 TidyBot++ 和 Shelf3D 做 real-to-sim-to-real 示例，展示 benchmark 中的状态、目标和规划流程可以落到真实物理交互中。

---

## 问题背景

### 要解决的问题

机器人在现实世界中行动时，需要同时处理自身运动学约束、物体几何关系、接触动力学、工具作用和任务目标。现有机器人学习 benchmark 常把这些问题混在感知、语言理解或具体应用场景里，导致很难判断一个方法失败是因为不会看图、不会理解语言，还是确实缺少 [[Physical Reasoning|物理推理]] 能力。

KinDER 的目标不是提出一个新策略网络，而是把物理推理拆成可控、可复现、可横向比较的任务分布：每个环境都尽量隔离某类物理约束，同时保留可程序生成的无限变化。

### 现有方法的局限

- [[Robot Learning Benchmark|机器人学习基准]] 往往偏向特定算法族，比如只测 IL/RL 或只测 foundation model agent，跨范式比较不足。
- 应用型 benchmark 任务更接近日常操作，但物理挑战、语言挑战、视觉挑战和场景常识高度耦合，难以定位瓶颈。
- 物理推理 benchmark 如 PHYRE 更偏 2D 抽象物理，和具身机器人操作里的技能、状态、接触、执行误差仍有距离。
- 强规划方法需要大量手工建模，学习方法需要大量数据，foundation model 方法常依赖语言或视觉先验，但它们在同一组物理任务上的相对强弱并不清楚。

### 本文动机

作者先回顾机器人规划/学习中反复出现但常被 one-off environment 处理的问题，再对照现有 benchmark 的空白，选择位于活跃研究前沿且现有 SOTA 不明确的挑战。这个设计让 KinDER 更像“物理推理试纸”：它不追求 household task 的完整复杂度，而是专门放大中层物理约束。

---

## 方法详解

### 总体结构

KinDER 由三层组成：

- **KinDERGarden**: 环境层。包含 Kinematic2D、Dynamic2D、Kinematic3D、Dynamic3D 四类环境，共 25 个环境，每个环境有程序生成变体。
- **KinDERGym**: 接口层。继承 [[Gymnasium]] API，提供 `reset()`、`step()`、对象中心 observation、固定维向量化 observation、参数化技能和 demonstrations。
- **KinDERBench**: 评测层。选取每类环境中代表性任务，统一运行 baseline，使用成功率、累计奖励、推理时间和工程成本分析方法。

### 五类核心挑战

- **基本空间关系**: 根据目标关系放置、移动或排列对象，例如把对象放到容器相对位置。
- **非抓取多物体操作**: 通过推、扫、拨等动作同时影响多个对象，例如把小物体扫入抽屉。
- **工具使用**: 使用钩子、盒子、扫具等中介改变可达性或力的传递方式。
- **组合几何约束**: 在箱子、柜子、架子等有限空间中同时满足多物体几何可行性。
- **动力学约束**: 任务成败取决于速度、碰撞、投掷、平衡等时间演化约束。

### 环境与状态设计

每个环境遵循标准 [[Markov Decision Process|MDP]] 形式，暴露 observation space、action space、初始状态分布和 `step()` 动态。奖励默认稀疏：未成功时每步为 -1，达到目标时终止。环境内部采用 [[Object-Centric Representation|对象中心状态]]，但也能把固定对象数的变体展平成固定维向量，方便 MLP、RL、DP 等方法使用。

这个对象中心设计有两个意图：一方面降低视觉感知噪声，让评测聚焦物理推理；另一方面保留对象数量变化，使 OOD 评估和 test-time scaling 可测。

### Baseline 谱系

KinDERBench 的 baseline 覆盖以下方法族：

- **规划方法**: BP、MPC、GSC、MBRL，代表显式模型、采样规划和学习动力学模型。
- **Foundation-model 规划**: LLMPlan、VLMPlan、LLMCon、VLMCon，使用 GPT-5.2，比较 zero-shot 与 in-context 示例，以及 object-centric state + RGB image 的增益。
- **强化学习**: PPO、SAC，代表 on-policy 与 off-policy continuous control。
- **模仿学习**: DP、DPES、finetuned VLA。DPES 额外输入环境状态，VLA 使用微调的 $\pi_{0.5}$，但训练/推理时不访问环境状态。

---

## 关键公式

### 公式1: [[Markov Decision Process|KinDER 环境交互]]

$$
s_{t+1}, o_{t+1}, r_t, d_t = \mathrm{step}(s_t, a_t), \qquad s_0 \sim p_0(s)
$$

**含义**: 将 KinDERGarden 的 Gymnasium 接口写成 MDP 交互过程；agent 在状态 $s_t$ 下执行动作 $a_t$，环境返回下一状态、观测、奖励和终止标记。

**符号说明**:
- $s_t$: 第 $t$ 步环境状态，通常包含机器人和对象的对象中心属性。
- $o_t$: agent 可见观测，可以是对象中心状态、向量化状态或 RGB 图像。
- $a_t$: 动作或参数化技能调用。
- $r_t$: 单步奖励。
- $d_t$: episode 是否终止。
- $p_0(s)$: 程序生成的初始状态分布。

### 公式2: [[Sparse Reward|稀疏奖励与累计奖励]]

$$
r_t =
\begin{cases}
0, & \text{if } g(s_t)=1,\\
-1, & \text{otherwise,}
\end{cases}
\qquad
R(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T-1} r_t \mid g(s_T)=1\right]
$$

**含义**: 论文使用稀疏奖励衡量成功 episode 的效率；越快达成目标，累计负奖励越接近 0。

**符号说明**:
- $g(s_t)$: 目标判定函数，成功时为 1。
- $\pi$: 被评估策略或规划器。
- $\tau$: 由策略和环境交互产生的轨迹。
- $T$: episode 长度。
- $R(\pi)$: 只在成功 episode 上统计的累计奖励均值。

### 公式3: [[Success Rate|成功率指标]]

$$
\mathrm{SR}(\pi)=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\left[g\left(s^{(i)}_{T_i}\right)=1\right]
$$

**含义**: KinDERBench 的主要有效性指标；每个 baseline 在多个随机种子和 episode 上评估，统计最终成功比例。

**符号说明**:
- $N$: 总评估 episode 数。
- $s^{(i)}_{T_i}$: 第 $i$ 条轨迹的终止状态。
- $\mathbf{1}[\cdot]$: 指示函数。

---

## 关键图表

### Figure 1: Core Challenges for Physical Reasoning in KinDER

![Figure 1](https://arxiv.org/html/2604.25788v1/x1.png)

**说明**: 该图把 KinDER 的五类核心物理推理挑战放在同一张图中：空间关系、非抓取多物体操作、组合几何约束、工具使用和动力学约束。它是理解整个 benchmark 设计边界的入口。

### Figure 2: KinDERGarden Core Challenges

![Figure 2](https://arxiv.org/html/2604.25788v1/x2.png)

**说明**: 展示 KinDERGarden 环境如何覆盖五类挑战，强调 benchmark 不是单一任务集合，而是按物理推理维度组织的环境谱系。

### Figure 3: Procedural Task Generation Example

![Figure 3](https://arxiv.org/html/2604.25788v1/x3.png)

**说明**: 以 `ConstrainedCupboard3D` 为例说明程序生成如何改变柜内隔板和杆件放置可行性，从而迫使 agent 推理组合几何约束。

### Figure 5: Real-to-Sim-to-Real Example

![Figure 5](https://arxiv.org/html/2604.25788v1/x5.png)

**说明**: 使用真实 TidyBot++ 的观测初始化 Shelf3D twin simulation，在仿真中生成动作计划，再回到真实机器人执行；该图支撑 KinDER 与真实物理交互之间的对应关系。

### Table II: 主 benchmark 平均成功率排序

| 方法 | 平均成功率 SR |
|------|---------------|
| BP | 0.57 |
| LLMCon | 0.43 |
| VLMCon | 0.43 |
| LLMPlan | 0.34 |
| VLMPlan | 0.34 |
| MPC | 0.32 |
| VLA | 0.32 |
| GSC | 0.26 |
| DPES | 0.25 |
| DP | 0.24 |
| PPO | 0.13 |
| MBRL | 0.08 |
| SAC | 0.02 |

**说明**: 显式工程化程度高的 BP 仍最强，但 LLM/VLM 加 in-context examples 后接近第二梯队；普通 RL 在稀疏奖励、长 horizon 下明显吃亏。

### Table III: SweepIntoDrawer3D 子任务成功率

| 方法 | Open Drawer | Grasp Sweeper | Sweep Some Objects | Sweep All Objects |
|------|-------------|---------------|--------------------|-------------------|
| DPES | 0.50 | 0.28 | 0.08 | 0.04 |
| DP | 0.87 | 0.74 | 0.22 | 0.14 |
| VLA | 0.01 | 0.00 | 0.00 | 0.00 |

**说明**: DP 在该长 horizon、多阶段任务上反而优于带环境状态的 DPES，提示“额外状态输入”并不自动转化为可用推理能力。

### Table IV: DynObstruction2D OOD 泛化

| Baseline | Train 1 obs | Test 0 obs | Test 2 obs | Test 3 obs |
|----------|-------------|------------|------------|------------|
| DP | 0.33 | 0.35 | 0.30 | 0.26 |
| VLA | 0.50 | 0.52 | 0.43 | 0.40 |

**说明**: VLA 在障碍数量变化时保持较稳健，作者推测可能来自预训练先验。

### Table V: StickButton2D 中 BP 的 test-time scaling

| 按钮数 | SR | Rwd | Inf-Time |
|--------|----|-----|----------|
| 1 | 0.99 | -52.60 | 1.51 |
| 3 | 0.26 | -86.30 | 19.45 |
| 5 | 0.02 | -123.80 | 28.12 |
| 10 | 0.00 | - | 38.39 |

**说明**: 随着对象数上升，规划成功率急剧下降、推理时间上升，暴露组合搜索在物理推理规模化上的瓶颈。

---

## 实验结果

### 设置

- **任务选择**: 从 Kinematic2D、Dynamic2D、Kinematic3D、Dynamic3D 四类中各选代表任务，包括 Motion2D、StickButton2D、DynObstruction2D、DynPushPullHook2D、BaseMotion3D、Transport3D、Shelf3D、SweepIntoDrawer3D。
- **评估规模**: 每个 baseline 使用 5 个随机种子，每个种子 50 个 evaluation episodes。
- **指标**: 成功率 SR、成功 episode 上的累计奖励 Rwd、每 episode wall-clock 推理时间 Inf-Time，并讨论工程成本。

### 主要发现

1. **BP 最强但工程成本最高**: BP 平均 SR 为 0.57，明显领先，但依赖手写技能和概念，迁移到新环境需要额外建模。
2. **In-context 示例帮助 LLM/VLM**: LLMCon/VLMCon 平均 SR 均为 0.43，高于 LLMPlan/VLMPlan 的 0.34，说明示例对 foundation-model planning 有实际收益。
3. **VLM 图像输入没有明显优势**: LLMPlan 与 VLMPlan、LLMCon 与 VLMCon 的结果接近，作者认为 VLM 未能有效利用额外 RGB 图像，尤其是在已有对象中心状态时。
4. **VLA 在部分动态/工具任务上意外强**: 在 DynPushPullHook2D 上，VLA 是唯一达到非平凡成功率的 baseline，SR 为 0.43；这很意外，因为 2D 渲染和物理与 VLA 预训练数据差异较大。
5. **DP 与 DPES 对比暴露状态利用问题**: DPES 有对象中心状态输入，但整体与 DP 接近，SweepIntoDrawer3D 上甚至低于 DP，说明模型未必能自动把结构化状态转成更好的长程操作策略。
6. **RL 在稀疏奖励下脆弱**: PPO/SAC 只在短 horizon 任务上表现尚可；附录中 dense reward 能改善 PPO，但总体成功率仍低，说明 KinDER 对 reward design 和 inductive bias 很敏感。
7. **MBRL 不如 MPC**: 两者使用同类 planner，但 MBRL 的学习转移模型不可靠，导致低于直接基于环境模型的 MPC。
8. **真实机器人验证是 proof-of-correspondence**: Shelf3D real-to-sim-to-real 使用 overhead camera 估计机器人和物体 pose，再初始化仿真规划并真实执行，证明环境抽象有现实对应，但还不是大规模真实评测。

---

## 批判性思考

### 优点

1. **问题切得准**: KinDER 有意识地把感知和语言难题降到较低，把 benchmark 焦点放在中层物理推理，对诊断规划/学习方法很有价值。
2. **跨范式比较完整**: 同时放入 TAMP、MPC、RL、IL、VLA、LLM/VLM，让结果能回答“哪类方法在哪类物理约束下更吃亏”。
3. **对象中心状态很适合分析**: 它降低视觉噪声，也让 LLM/VLM、DP/DPES 等对照更干净，可以直接看结构化状态是否被利用。
4. **开源和 Gymnasium 接口降低复现门槛**: `pip install kindergarden`、演示数据和 baseline 仓库让后续方法较容易接入。

### 局限性

1. **仿真物理仍是抽象**: 作者承认真实接触、摩擦、柔性、传感噪声等细粒度物理没有完全覆盖；KinDER 更像中层推理 benchmark，不是完整机器人现实世界 benchmark。
2. **观测假设偏干净**: 对象中心状态让评测更清晰，但也可能高估某些规划方法、低估视觉端到端方法在真实视觉场景中的综合能力。
3. **排除因素较多**: 当前范围不重点覆盖随机性、部分可观测、多机器人、多 embodiment 等现实机器人重要因素。
4. **baseline 代表性有限**: 13 个 baseline 覆盖面很广，但仍没有评估很多近年的 hybrid planner、world-model agent、test-time search + learned proposal 等方法。
5. **项目页与论文 baseline 数有小不一致**: 论文摘要和正文写 13 个 baseline，项目页 About 段落写 8 个 implemented baselines，后续引用时应以论文正文为准并留意版本更新。

### 潜在改进方向

1. **学习增强规划**: 用学习模型提出候选技能参数、剪枝组合搜索，缓解 StickButton2D 里 BP 随对象数扩展失败的问题。
2. **状态-视觉双通道诊断**: 系统比较 object-centric state、RGB、depth、segmentation、noisy state，分离表示误差与物理推理误差。
3. **真实机器人小规模 suite**: 将 Shelf3D 扩展成多个真实任务，形成 sim benchmark 与 real benchmark 的配对子集。
4. **更强 foundation-model agent**: 评估带工具调用、反思、失败恢复、物理仿真查询的 LLM/VLM planner，而不是只看 open-loop skill sequence。
5. **动态约束专门化指标**: 除成功率外增加接触质量、动量利用、碰撞安全、约束余量等指标，避免所有动态推理都被压缩成最终 success。

### 可复现性评估

- [x] 代码开源：kindergarden 与 kinder-baselines 均公开。
- [x] 环境接口清楚：Gymnasium-compatible，含 object-centric state 和 vectorization。
- [x] baseline 覆盖广：规划、RL、IL、FM 都有实现。
- [x] 评估协议明确：5 seeds，每 seed 50 episodes，报告 SR/Rwd/Inf-Time。
- [ ] 真实机器人验证有限：目前更像示例，不是完整 real-world benchmark。
- [ ] 依赖外部大模型：GPT-5.2、$\pi_{0.5}$ 等模型版本会影响长期可复现性。

---

## 关联笔记

- [[Physical Reasoning]]
- [[Robot Learning Benchmark]]
- [[Task and Motion Planning]]
- [[Gymnasium]]
- [[Object-Centric Representation]]
- [[Model Predictive Control]]
- [[Diffusion Policy]]
- [[Vision-Language-Action Model]]
- [[Reinforcement Learning]]
- [[Sim-to-Real]]
- [[Sparse Reward]]
- [[Success Rate]]
