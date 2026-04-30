---
title: "GigaWorld-Policy: An Efficient Action-Centered World--Action Model"
method_name: "GigaWorld-Policy"
authors:
  - GigaAI
  - Boyuan Wang
  - Chaojun Ni
  - Guan Huang
  - Guosheng Zhao
  - Hao Li
  - Hengtao Li
  - Jie Li
  - Jindi Lv
  - Jingyu Liu
  - Min Cao
  - Peng Li
  - Qiuping Deng
  - Wenjun Mei
  - Xiaofeng Wang
  - Xinze Chen
  - Xinyu Zhou
  - Yang Wang
  - Yifan Chang
  - Yifan Li
  - Yukun Zhou
  - Yun Ye
  - Zhichao Liu
  - Zheng Zhu
year: 2026
venue: arXiv
arxiv: "2603.17240"
paper_url: "https://arxiv.org/abs/2603.17240"
pdf_url: "https://arxiv.org/pdf/2603.17240"
project_url: "https://gigaai-research.github.io/GigaWorld-Policy/"
tags: [robot-policy, world-action-model, video-generation, flow-matching, visuomotor-control, embodied-ai]
image_source: online
created: 2026-04-29
---

# GigaWorld-Policy

## 元信息

- 论文：GigaWorld-Policy: An Efficient Action-Centered World--Action Model
- 方法名：GigaWorld-Policy
- 链接：<https://arxiv.org/abs/2603.17240>
- 项目页：<https://gigaai-research.github.io/GigaWorld-Policy/>
- 备注：本文以 `https://arxiv.org/abs/2603.17240` 为准。

## 一句话

GigaWorld-Policy 把 [[World-Action Model|世界-动作模型]] 的中心从“同时生成未来视频和动作”改成“先高效预测动作、训练时用未来视频约束动作合理性”，从而在部署时可以跳过显式视频生成，获得更快的机器人策略推理。

## 核心贡献

- 提出 action-centered [[World-Action Model|WAM]]：动作 token 只看当前观测和机器人状态，未来视频 token 可以看动作，但动作不会被未来视频反向影响。
- 将策略学习拆成两个耦合任务：预测未来动作序列，同时以预测动作和当前观测为条件生成未来视频，用视频动态提供训练信号。
- 构建大规模 embodied video/data 预训练集，用 action-centered video generation backbone 迁移到机器人策略学习。
- 推理时默认只解码动作，不生成未来视频；相对 Motus，在真实机器人任务上约 9x 更快且成功率提高约 7 个百分点。
- 在 RoboTwin 2.0 上相比 $\pi_{0.5}$ 报告约 95% 的性能提升，并展示真实 PiPER 双臂任务与数据效率实验。

## 背景与问题

传统 [[World Model|世界模型]] 对机器人策略的吸引力在于：如果模型能预测未来视觉结果，它就可能学到动作的物理后果。但近期 [[World-Action Model|World-Action Model]] 往往把未来视觉 token 和动作 token 联合建模，带来两个问题：

- 推理慢：每次动作预测都伴随未来视频推理，视觉生成成本高。
- 表征纠缠：动作预测容易依赖未来视频质量；视频生成错误会污染 motion/action 表征。

GigaWorld-Policy 的关键判断是：视频生成适合作为训练时的物理约束，但不必成为部署时的必要路径。也就是说，未来视频是 teacher signal，而不是 runtime dependency。

## 方法

### 任务形式化

给定当前多视角观测 $O_t$、机器人状态 $s_t$ 和语言指令 $l$，模型需要预测未来动作序列 $A_{t:t+H-1}$。论文将策略写成动作条件分布：

[[Robot Policy|动作策略]]

$$
p_{\theta}\!\left(A_{t:t+H-1}\mid O_t, s_t, l\right)
$$

其中 $H$ 是动作预测 horizon，$O_t$ 是当前视觉观测，$s_t$ 是本体状态，$l$ 是任务语言，$A_{t:t+H-1}$ 是未来连续动作序列。GigaWorld-Policy 的部署路径只需要这个动作分布，而不强制输出未来帧。

### Action-Centered WAM

模型仍保留未来视频生成任务，但把依赖方向改成动作中心：

[[Conditional Video Generation|动作条件未来视频生成]]

$$
p_{\theta}\!\left(O_{t+1:t+K}\mid O_t, s_t, l, A_{t:t+H-1}\right)
$$

这里 $K$ 是未来视频帧数。该项的含义是：训练时要求预测动作能够解释可见的未来视觉变化，从而约束动作具备物理可行性；推理时可以不采样 $O_{t+1:t+K}$。

### Token 与注意力结构

论文把输入组织为四类 token：

- $T_o$：当前观测 token，来自多视角图像编码。
- $T_s$：机器人 proprioceptive state token。
- $T_a$：未来动作 token。
- $T_f$：未来视频 token。

核心注意力 mask 是单向因果设计：$T_a$ 只能 attend 到 $T_o$ 与 $T_s$，而 $T_f$ 可以 attend 到 $T_o,T_s,T_a$。因此动作预测不会读取未来视频 token，避免训练阶段的视频生成分支泄漏到动作分支。

## 关键公式

### Flow Matching 训练目标

GigaWorld-Policy 使用 [[Flow Matching|flow matching]] 风格的连续噪声到数据路径。对动作 latent $z_1^a$ 与噪声 $z_0^a\sim\mathcal{N}(0,I)$，中间状态可以写作：

[[Flow Matching|线性插值路径]]

$$
z_{\tau}^{a}=(1-\tau)z_0^{a}+\tau z_1^{a},\qquad
u_{\tau}^{a}=z_1^{a}-z_0^{a},\qquad \tau\in[0,1]
$$

其中 $z_{\tau}^{a}$ 是时间 $\tau$ 的 noisy action latent，$u_{\tau}^{a}$ 是目标速度场。动作分支学习速度场 $v_{\theta}^{a}$：

[[Action Prediction Loss|动作流匹配损失]]

$$
\mathcal{L}_{a}
=\mathbb{E}_{\tau,z_0^a,z_1^a}
\left[
\left\|
v_{\theta}^{a}\!\left(z_{\tau}^{a},\tau\mid O_t,s_t,l\right)-u_{\tau}^{a}
\right\|_2^2
\right]
$$

视频分支采用同构目标，只是条件额外包含动作 latent：

[[Video Prediction Loss|视频流匹配损失]]

$$
\mathcal{L}_{f}
=\mathbb{E}_{\tau,z_0^f,z_1^f}
\left[
\left\|
v_{\theta}^{f}\!\left(z_{\tau}^{f},\tau\mid O_t,s_t,l,z_1^a\right)
-
\left(z_1^f-z_0^f\right)
\right\|_2^2
\right]
$$

总训练目标为：

[[Multi-Task Learning|联合训练目标]]

$$
\mathcal{L}
=\lambda_a\mathcal{L}_a+\lambda_f\mathcal{L}_f
$$

其中 $\lambda_a,\lambda_f$ 控制动作监督与未来视频监督的权重。直观上，$\mathcal{L}_a$ 保证策略能执行任务，$\mathcal{L}_f$ 迫使动作 latent 能解释未来视觉动态。

### 推理路径

推理时只运行动作分支，并通过速度场积分得到动作 latent：

[[ODE Sampling|动作采样]]

$$
z_1^a=z_0^a+\int_0^1
v_{\theta}^{a}\!\left(z_{\tau}^{a},\tau\mid O_t,s_t,l\right)\,d\tau
$$

然后由动作 decoder 得到最终连续控制序列。未来视频分支可用于诊断、可视化或需要世界预测的场景，但不是实时控制的必需项。

## 关键图表

![Figure 1: inference frequency and task success comparison](https://arxiv.org/html/2603.17240v2/x1.png)

Figure 1 把 GigaWorld-Policy 放在“推理频率 vs. 真实任务成功率”的二维图里，与 Motus、Cosmos-Policy、$\pi_{0.5}$ 等对比。它传达的核心结果是：GigaWorld-Policy 在保持或提升成功率的同时，把 WAM 类方法的部署频率显著推高。

![Project teaser: architecture and rollout examples](https://gigaai-research.github.io/GigaWorld-Policy/static/images/pipeline.png)

项目页的 pipeline 图展示了训练时 action prediction 与 future video generation 两个分支如何耦合，以及部署时如何只保留动作解码路径。

![Figure 4: action-centered attention mask](https://arxiv.org/html/2603.17240v2/x4.png)

Figure 4 是方法最关键的结构图：动作 token 不接收 future-video token 信息，future-video token 接收动作信息。这一点是“训练时用视频、推理时可跳过视频”的结构前提。

## 实验结果

### 推理速度

- 论文报告 GigaWorld-Policy 在 NVIDIA A100 上相对 Motus 约 9x 更快。
- 相比需要显式联合推理未来视频与动作的 WAM，GigaWorld-Policy 的动作中心设计减少了部署时 token 数和视频生成开销。

### 仿真基准

- 在 RoboTwin 2.0 上，论文报告 GigaWorld-Policy 相比 $\pi_{0.5}$ 有约 95% 性能提升。
- 这部分结果支持作者的主张：先通过大规模 embodied data 预训练 action-centered world-action backbone，再迁移到机器人策略，能提升复杂 manipulation benchmark 的泛化。

### 真实机器人

- 真实环境使用 PiPER 机械臂，包含 QR code scanning、trash sweeping 等任务。
- 论文报告真实任务上 GigaWorld-Policy 相比领先 WAM baseline Motus 成功率提升约 7 个百分点，同时推理频率显著更高。
- 数据效率实验显示，GigaWorld-Policy 用 10% 真实任务数据即可接近或匹配 $\pi_{0.5}$ 的完整数据表现。

### 消融

论文的 ablation 关注 action-centered 结构、未来视频辅助监督与数据规模。结论倾向于：

- 去掉未来视频监督会削弱动作的物理一致性。
- 让动作 token 依赖 future-video token 会重新引入推理依赖与表征纠缠。
- 大规模 embodied video pretraining 对下游策略成功率有明显贡献。

## 批判性思考

- 优点：方法抓住了 WAM 部署慢的真实瓶颈，把视频生成从 runtime requirement 降级为 training regularizer；这是工程上很干净的取舍。
- 优点：注意力 mask 的设计使“可选视频生成”不是口头承诺，而是结构保证；动作分支天然可以独立运行。
- 风险：论文的核心收益依赖大规模 embodied data 预训练，复现门槛较高；数据组成、清洗策略和训练预算会显著影响结论。
- 风险：未来视频作为训练约束是否真正提升物理因果理解，还是只提供更强的视觉表征正则，还需要更细粒度诊断实验。
- 风险：真实任务展示仍集中在相对短 horizon 的桌面 manipulation；对长程、多阶段、强接触和失败恢复任务的泛化尚不充分。
- 观察：GigaWorld-Policy 与 $\pi_{0.5}$ 的对比很有价值，但两者在数据、模型初始化、动作表示和部署频率上的差异较多，最好进一步做 compute/data-matched 对照。

## 关联笔记

- [[World-Action Model]]
- [[World Model]]
- [[Robot Policy]]
- [[Flow Matching]]
- [[Conditional Video Generation]]
- [[Video Prediction]]
- [[Visuomotor Policy]]
- [[RoboTwin]]
- [[PiPER]]
- [[pi0.5]]
- [[Motus]]
- [[Cosmos-Policy]]

## 自检

- Frontmatter：已包含 title、method_name、authors、year、venue、arxiv、url、tags、image_source、created。
- 公式：包含 6 个 LaTeX display math blocks，覆盖策略分布、视频条件分布、flow matching 路径、动作损失、视频损失、联合损失、ODE 采样。
- 图片：包含 3 个在线图片外链，其中至少 2 个来自 arXiv HTML，1 个来自项目页。
- 关键 section：元信息、一句话、贡献、背景、方法、关键图表、实验结果、批判性思考、关联笔记均已包含。
- 限制：根据“只写入目标文件”的要求，未创建概念页、未刷新 MOC、未下载/本地化图片。
