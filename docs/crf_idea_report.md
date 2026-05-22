# Action Response Field World Model 研究想法汇报

## 1. 一句话核心想法

现有 World Action Model 主要预测“执行某个动作后未来会变成什么”，但机器人控制更需要知道“如果动作稍微改变，未来会如何改变”。因此，我们提出 **Action Response Field / Causal Response Field World Model**：显式建模动作扰动对未来交互状态的局部响应，用于提升接触敏感任务中的动作修正、闭环控制和鲁棒性。

核心主张：

```text
World models should not only predict what will happen after an action,
but also predict how the outcome would change if the action changed.
```

中文表达：

```text
世界模型不应只是未来生成器，还应成为动作-世界响应算子。
```

## 2. 研究背景

近年来具身智能中的世界模型和视觉语言动作模型发展很快。典型范式包括：

- VLA：输入图像和语言指令，直接输出机器人动作。
- WAM：输入当前观测和动作，预测未来视频、未来状态或动作轨迹。
- FastWAM：保留视频和动作联合训练，但测试时跳过显式未来想象，直接快速输出动作。

这些方法已经证明了大规模视觉生成模型和动作模型对机器人控制有帮助，但它们仍然有一个共同限制：

**它们通常只学习一个动作对应的结果，而没有显式学习动作变化对结果变化的影响。**

例如在 Push-T、推物体、开门、插孔、布料对齐等任务中，机器人动作稍微偏一点，未来结果可能完全不同。普通 WAM 可以预测当前动作的未来，但如果预测结果不够好，模型本身并不会直接告诉我们动作应该往哪个方向微调。

## 3. 现有 WAM 的不足

普通 action-conditioned WAM 学习的是：

```text
s_t, a_t -> s_{t+h}
```

它回答的问题是：

```text
给定当前状态和动作，未来会是什么？
```

但机器人控制常常更关心：

```text
如果动作 a_t 改成 a_t + δa，未来 s_{t+h} 会怎样变化？
```

也就是：

```text
δa -> δs_{t+h}
```

这个问题在接触丰富和精细操作任务中特别重要。

举例：

- 推 T 形块时，接触点上移可能增加旋转，接触点右移可能增加平移。
- 开抽屉时，拉力方向稍微偏斜可能导致卡住。
- 插孔时，动作角度微小变化可能决定成功或失败。
- 布料整理时，抓取点和拉动方向的微小差别会导致不同褶皱传播。

现有 WAM 通常能预测一个结果，但缺少对这些局部动作敏感性的显式建模。

## 4. 核心定义：Action Response Field

我们希望学习一个响应算子：

```text
R_theta(s_t, a_t): δa -> δs_{t+h}
```

或者用 Jacobian 的形式表示：

```text
J_theta(s_t, a_t) = ∂s_{t+h} / ∂a_t
```

其中：

- `s_t` 是当前视觉状态、latent state 或交互状态；
- `a_t` 是当前候选动作；
- `s_{t+h}` 是未来状态；
- `δa` 是动作扰动；
- `δs_{t+h}` 是该动作扰动导致的未来变化。

训练目标可以写成：

```text
s_{t+h}(a_t + δa) - s_{t+h}(a_t) ≈ R_theta(s_t, a_t) δa
```

也就是说，模型不是只预测某个动作的未来，而是预测 **动作变化会如何改变未来**。

## 5. 和普通 WAM 的区别

普通 WAM：

```text
输入：当前状态 + 动作
输出：未来状态 / 未来视频
```

Action Response Field：

```text
输入：当前状态 + 候选动作
输出：动作扰动到未来变化的响应关系
```

普通 WAM 关心：

```text
What will happen?
```

Action Response Field 关心：

```text
How would the future change if the action changed?
```

因此，两者的控制意义不同：

- WAM 给出一个预测结果。
- CRF 给出一个动作修正方向。

例如 FastWAM 输出一个初始动作 `a0`。如果该动作会让物体偏离目标，普通模型通常只能重新采样其他动作再试。CRF 则希望直接估计：

```text
为了让未来更接近目标，动作应该朝哪个方向微调。
```

## 6. 和 Inverse Dynamics 的区别

Inverse Dynamics 学习：

```text
s_t, s_{t+h} -> a_t
```

它回答：

```text
从当前状态到目标未来，大概应该执行什么动作？
```

这更像一种目标条件行为克隆，直接给出动作答案。

CRF 学习：

```text
s_t, a_t, δa -> δs_{t+h}
```

它回答：

```text
如果当前动作这样改，未来会怎样变？
```

因此：

- Inverse Dynamics 是“给动作”。
- CRF 是“告诉动作怎么改”。

如果 inverse dynamics 给错了动作，它本身通常不提供局部修正信息。CRF 则可以用于迭代优化：

```text
初始动作 a0
 -> 预测未来误差 e
 -> CRF 估计动作修正 δa
 -> 得到新动作 a1 = a0 + δa
```

## 7. 和 MPC 的关系

MPC 的本质是：

```text
当前状态
 -> 在脑内尝试多条候选动作序列
 -> 用动力学模型预测未来
 -> 选择代价最低的动作
 -> 只执行第一步
 -> 下一帧重新观测和规划
```

WAM 可以作为 MPC 的 learned dynamics model：

```text
s_t, a_{t:t+H} -> s_{t+H}
```

但是普通 WAM + MPC 往往需要大量采样和 rollout。CRF 的作用是把局部优化信息显式学出来：

```text
δs_future ≈ J δa
```

如果当前预测未来和目标之间有误差：

```text
e = s_goal - s_pred
```

那么可以求一个动作修正：

```text
δa* = argmin ||J δa - e||^2
```

然后更新动作：

```text
a_new = a_old + δa*
```

因此，CRF 可以看作一种 learned visual servoing / local MPC update。它不一定替代 MPC，而是可以让动作搜索更高效、更有方向。

## 8. 初期定位：FastWAM 的动作修正插件

为了降低工程风险，第一阶段不从零训练一个新 WAM，而是在已经跑通的 FastWAM 上增加一个轻量动作响应模块。

初始流程：

```text
observation + language instruction
        ↓
FastWAM 输出初始动作 a0
        ↓
Action Response Field 估计动作响应 R(s, a0)
        ↓
根据目标误差或任务进展计算动作修正 δa
        ↓
执行修正动作 a0 + δa
```

这可以被理解为一个可插拔的 controllability layer。

但论文最终可以更一般地表述为：

```text
We propose an action-response modeling objective that can be added to WAM/VLA backbones,
turning direct action prediction into controllability-aware action generation.
```

也就是说，工程上先作为插件验证，论文上可以扩展为 response-aware WAM 框架。

## 9. 为什么不是简单后处理

普通后处理可能只是：

- 平滑动作；
- 缩放动作；
- 加噪声采样；
- 用 value model rerank；
- 用 success classifier 判断好坏。

CRF 的不同点在于它学习的是结构化关系：

```text
动作扰动 -> 未来交互变化
```

它不是只判断一个动作好不好，而是估计：

```text
动作的哪一维应该怎么改，未来哪个区域会因此改变。
```

因此它更接近一个控制意义上的 response operator，而不是普通动作滤波器。

## 10. 训练数据选择

第一阶段建议优先使用仿真数据。

原因是 CRF 最需要的数据形式是：

```text
同一个初始状态 s
执行动作 a
执行动作 a + δa
比较两个未来状态差异
```

仿真中可以 reset 到完全相同状态，反复执行动作扰动。真实机器人难以做到完全同状态重置，数据成本也高。

训练路线可以分三步：

### 10.1 仿真中的理想监督

在 LIBERO、RoboTwin、Push-T、ManiSkill 等环境中，对同一状态采样多个动作扰动：

```text
(s, a, s')
(s, a + δa, s'_δ)
```

监督：

```text
s'_δ - s' ≈ R(s, a) δa
```

### 10.2 离线数据中的近邻监督

如果没有同状态多动作数据，可以在离线 demonstrations 中找相似状态：

```text
s_i ≈ s_j
a_i != a_j
s'_i != s'_j
```

近似构造：

```text
Δa = a_j - a_i
Δs = s'_j - s'_i
```

监督：

```text
Δs ≈ R(s_i, a_i) Δa
```

### 10.3 用已有 WAM 合成扰动数据

还可以先用 action-conditioned WAM 生成不同动作下的未来，再蒸馏一个 response head。但这一步要小心，因为合成数据会继承 WAM 的错误。

## 11. 预计在哪些任务提升最大

CRF 最适合以下任务：

### 11.1 接触敏感任务

例如：

- Push-T；
- block pushing；
- drawer opening；
- door opening；
- tool-use pushing / pulling；
- peg insertion 前的对准。

这些任务中，动作小扰动会显著改变物体运动结果，CRF 能提供动作微调方向。

### 11.2 精细操作任务

例如：

- 插孔；
- 旋钮；
- 按按钮；
- 夹取薄片；
- 狭窄空间放置。

这些任务的难点不是语义理解，而是动作精度。CRF 可以提升局部控制精度。

### 11.3 长程任务中的低层可靠性

CRF 本身不是长程任务规划器，但可以降低每一步低层动作误差：

```text
每一步动作更稳
 -> 误差累积更少
 -> 长程成功率提升
```

因此它对长程任务的作用是提升 low-level execution reliability，而不是负责 high-level planning。

## 12. 不建议第一阶段主打的方向

第一阶段不建议主打：

- 开放词汇零样本泛化；
- 复杂语言任务分解；
- 真实世界大规模部署；
- 完整 4D 世界生成；
- 超长程多阶段规划。

这些方向问题太大，容易稀释贡献。

更清晰的定位是：

```text
Contact-rich manipulation under local action sensitivity.
```

也就是：

```text
接触敏感任务中的动作响应建模与动作修正。
```

## 13. 技术实现草案

### 13.1 模型输入

可以使用：

```text
当前观测 latent z_t
语言 instruction embedding
proprio state
FastWAM 初始动作 a0
```

### 13.2 模型输出

第一版可以输出低维 response：

```text
R_theta(z_t, a0)
```

让它预测：

```text
未来 latent delta
```

之后可以扩展为空间 response field：

```text
每个视觉 patch / latent token 对动作扰动的响应强度和方向
```

### 13.3 Loss

核心 loss：

```text
L_response = || Δz_future - R_theta(z_t, a_t) Δa ||^2
```

其中：

```text
Δa = a_j - a_i
Δz_future = z_future_j - z_future_i
```

可选 regularization：

```text
L_sparse：鼓励响应集中在交互区域
L_smooth：鼓励相似状态下 response 连续
L_rank：鼓励 response operator 低秩，提升稳定性
```

## 14. 实验设计

### 14.1 离线诊断实验

目标：证明 CRF 确实学到了动作扰动和未来变化的关系。

指标：

- response prediction MSE；
- predicted future delta 与真实 future delta 的 cosine similarity；
- top-k response region localization；
- action perturbation direction accuracy。

### 14.2 动作修正实验

流程：

```text
FastWAM 输出动作 a0
CRF 估计修正 δa
执行 a0 + δa
```

对比：

- FastWAM；
- FastWAM + random perturbation；
- FastWAM + action smoothing；
- FastWAM + inverse dynamics correction；
- FastWAM + CRF。

指标：

- success rate；
- final distance to goal；
- number of replans；
- robustness under action noise；
- contact-sensitive failure rate。

### 14.3 泛化实验

测试：

- 新物体位置；
- 新目标姿态；
- 新摩擦或质量；
- 动作噪声；
- 视觉扰动；
- 未见过布局。

重点证明 CRF 学到的是动作-世界响应规律，而不只是记忆 demonstration。

## 15. 和 FastWAM 的关系

FastWAM 可以作为：

- strong baseline；
- action proposer；
- backbone；
- ablation 中的 `without response modeling`。

我们的第一阶段方法可以称为：

```text
FastWAM + Action Response Field
```

如果实验有效，第二阶段可以把 response objective 融入 FastWAM 训练，使其成为：

```text
Response-aware World Action Model
```

## 16. 预期创新点

### 创新点 1：从结果预测转向响应预测

现有 WAM 主要预测未来状态，我们显式预测动作扰动对未来状态的影响。

### 创新点 2：提出 Action Response Field

定义机器人视觉世界中的动作响应场，描述局部交互区域对动作变化的敏感性。

### 创新点 3：用于动作修正而非单纯生成

CRF 不只是生成视频或动作，而是为动作优化提供方向信息。

### 创新点 4：适合接触敏感任务

该方法特别针对 contact-rich manipulation 中动作微小变化导致结果显著变化的问题。

## 17. 主要风险和应对

### 风险 1：真实数据中缺少同状态多动作监督

应对：

- 第一阶段用仿真构造扰动数据；
- 第二阶段用离线近邻状态近似；
- 第三阶段再做少量真机验证。

### 风险 2：被质疑普通 WAM 可微，直接求梯度即可

应对：

- diffusion/video WAM 的梯度计算昂贵且噪声大；
- 普通未来预测 loss 不保证动作响应准确；
- 我们显式监督 `δa -> δfuture`，评估 response accuracy；
- 对比 baseline：gradient-through-WAM、sampling MPC、random perturbation。

### 风险 3：被认为只是动作后处理

应对：

- 强调我们学习的是动作响应算子，而非动作滤波；
- 设计 response prediction 指标；
- 可视化 action-response field；
- 展示其能预测动作维度和未来区域之间的对应关系。

## 18. 当前最小可行路线

### Step 1：确认 FastWAM 数据结构

已新增脚本：

```text
scripts/inspect_crf_batch.py
```

用于检查：

- video shape；
- action shape；
- proprio shape；
- action/video 时间对齐；
- VAE latent shape。

### Step 2：构造离线 response pair

从 batch 或 dataset 中构造：

```text
(z_i, a_i, z'_i)
(z_j, a_j, z'_j)
```

其中 `z_i` 和 `z_j` 尽量相似。

### Step 3：训练轻量 response head

第一版不改 FastWAM 主干，只训练：

```text
ResponseHead(z_t, a_t, Δa) -> Δz_future
```

### Step 4：离线评估 response accuracy

先不跑闭环控制，只验证：

```text
模型能否预测动作扰动导致的未来 latent 变化。
```

### Step 5：接入动作修正

使用 FastWAM 输出初始动作，再用 CRF 修正：

```text
a_new = a_fastwam + δa
```

## 19. 给导师汇报时的推荐表述

可以这样介绍：

```text
我们观察到，现有 WAM/VLA 通常只预测动作结果或直接生成动作，
但机器人控制中更关键的是知道动作该如何微调。
因此我们尝试提出 Action Response Field，
显式学习动作扰动对未来交互状态的影响。
它可以作为 FastWAM 等强 WAM 模型上的可插拔 controllability layer，
用于提升接触敏感任务中的动作精度和鲁棒性。
```

更短版本：

```text
FastWAM tells the robot what action to take.
Our Action Response Field tells the robot how the outcome would change if that action changed,
therefore enabling controllability-aware action refinement.
```

## 20. 暂定论文题目

可选题目：

```text
Action Response Fields: Learning How Robot Actions Locally Change the World
```

```text
From Future Prediction to Action Response: Controllability-Aware World Action Models
```

```text
Action-Response World Models for Contact-Rich Robot Manipulation
```

```text
Learning Local Action Sensitivity for World-Model-Based Robot Control
```

