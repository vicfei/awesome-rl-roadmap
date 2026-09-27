# 贝尔曼方程完整推导（配套讲义）

> [Awesome RL Roadmap](../README.zh-CN.md) 配套推导 · 对应 [Stage 1](../README.zh-CN.md#stage-1--经典-rl-基础12-周)。
> 覆盖贝尔曼期望方程与贝尔曼最优方程的完整推导。
>
> **全文假设有限 MDP**（状态空间与动作空间均为有限集）：这保证下文所有 $\max$、$\arg\max$ 良定义。连续动作空间需将 $\max$ 换为 $\sup$ 并另行讨论可测性与可达性。

## 一、先定义清楚基本概念

### 1. 马尔可夫决策过程（MDP）

一个 MDP 由五元组 $ (S, A, P, R, \gamma) $ 描述：

- $ S $：状态空间。
- $ A $：动作空间。
- $ P(s'|s,a) $：在状态 $ s $ 执行动作 $ a $ 后，转移到状态 $ s' $ 的概率。
- $ R(s,a,s') $：在状态 $ s $ 执行动作 $ a $，转移到 $ s' $ 时获得的即时奖励。有时简写为 $ R(s,a) $。
- $ \gamma \in [0,1] $：折扣因子。

### 2. 策略

策略 $ \pi(a|s) $ 表示在状态 $ s $ 下选择动作 $ a $ 的概率。本文中的 $ \pi $ 一律指**平稳的马氏策略**：动作分布只依赖当前状态，不随时间变化、不依赖历史。

### 3. 回报（Return）

从时刻 $ t $ 开始的折扣累积回报定义为：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

它有一个关键的递归形式：

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

**为什么会有 $ \gamma $？**

- 数学上：如果任务不终止且奖励恒正，不折扣的无穷级数会发散。$ \gamma < 1 $ 保证级数收敛。
- 直觉上：未来的奖励不如现在的奖励值钱，$ \gamma $ 表达这种偏好。
- 递归式不是额外加上 $ \gamma $，而是把定义中本来就存在的折扣结构提取出来。

### 4. 状态价值函数

在策略 $ \pi $ 下，状态 $ s $ 的价值定义为从 $ s $ 出发的期望回报：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
$$

### 5. 动作价值函数

在策略 $ \pi $ 下，状态 $ s $ 执行动作 $ a $ 的价值：

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
$$

---

## 二、贝尔曼期望方程的推导

### 第 1 步：从定义出发，拆开回报

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
$$

利用 $ G_t = R_{t+1} + \gamma G_{t+1} $：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right]
$$

### 第 2 步：利用期望的线性性

期望是线性的，可以拆成两项：

$$
V^\pi(s) = \underbrace{\mathbb{E}_\pi \left[ R_{t+1} \mid S_t = s \right]}_{\text{即时奖励的期望}} + \gamma \underbrace{\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]}_{\text{未来回报的期望}}
$$

### 第 3 步：计算即时奖励的期望

在状态 $ s $ 下，动作 $ A_t $ 按策略 $ \pi(a|s) $ 选择，然后环境转移到 $ s' $，给出奖励 $ R(s,a,s') $。用全期望公式，先对动作展开，再对下一状态展开：

$$
\mathbb{E}_\pi \left[ R_{t+1} \mid S_t = s \right]
=
\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')
$$

如果奖励只依赖 $ (s,a) $，可以简写为 $ \sum_a \pi(a|s) R(s,a) $。

### 第 4 步：计算未来回报的期望（详细展开）

这是最容易被含糊带过的一步。我们把它拆成三次展开。

#### 4.1 先对动作 $ A_t $ 展开

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]
=
\sum_a \underbrace{P(A_t = a \mid S_t = s)}_{\pi(a|s)} \cdot
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
$$

所以：

$$
= \sum_a \pi(a|s) \cdot \mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
$$

#### 4.2 再对下一状态 $ S_{t+1} $ 展开

对于固定的动作 $ a $，环境会随机转移到 $ s' $，概率为 $ P(s'|s,a) $。再用一次全期望公式：

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
=
\sum_{s'} P(s'|s,a) \cdot
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \right]
$$

#### 4.3 利用马尔可夫性简化条件期望（注意这里实际用了三个前提）

要证明

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \right]
=
\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right]
$$

需要的不只是"环境马尔可夫"，而是三个前提共同作用：

1. **环境马尔可夫性**：给定 $ S_{t+1}=s' $，环境的未来转移与奖励与历史 $ (S_t, A_t) $ 无关；
2. **策略平稳且马氏**：$ t+1 $ 时刻之后的动作由 $ \pi $ 只根据彼时的状态生成，同样与历史无关——所以条件里多出来的 $ (S_t=s, A_t=a) $ 不改变未来动作的分布；
3. **时齐性**：$ P $、$ R $ 与 $ \pi $ 不随时间变化，因此

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] = V^\pi(s')
$$

右边正是 $ V^\pi $ 的定义在时刻 $ t+1 $ 的取值——时齐性保证它是同一个函数，无需另记 $ V^\pi_{t+1} $。

#### 4.4 合并得到未来回报的期望

把 4.2 和 4.3 代回 4.1：

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]
=
\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')
$$

### 第 5 步：合并两项，得到贝尔曼期望方程

把第 3 步和第 4 步的结果代回第 2 步：

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')
+ \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')
$$

合并求和：

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

如果奖励只依赖 $ (s,a) $，则：

$$
V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
$$

这就是**贝尔曼期望方程**。

### 第 6 步：动作价值函数的贝尔曼期望方程

类似地，从 $ Q^\pi(s,a) $ 的定义出发：

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
$$

拆开回报：

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \right]
$$

即时奖励的期望是 $ \sum_{s'} P(s'|s,a) R(s,a,s') $。未来回报对 $ s' $ 展开（与 4.2、4.3 相同的论证）：

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
=
\sum_{s'} P(s'|s,a) V^\pi(s')
$$

所以：

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

又因为 $ V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a) $，两个方程可以互相代入。

---

## 三、贝尔曼最优方程的推导

### 第 1 步：定义最优价值函数

$$
V^*(s) = \max_\pi V^\pi(s)
$$

它表示在所有可能的策略中，从状态 $ s $ 出发能获得的最大期望回报。类似地：

$$
Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

### 第 2 步：最优策略的存在性与贪心性质

有限 MDP 中可以证明：**存在一个确定性最优策略 $ \pi^* $，它在所有状态同时达到 $ V^* $**（证明思路：贝尔曼最优算子是 $ \gamma $-压缩映射，$ V^* $ 存在且唯一；对 $ V^* $ 贪心的策略达到它）。于是它在每个状态选择使 $ Q^*(s,a) $ 最大的动作：

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
\qquad\text{因此}\qquad
V^*(s) = \max_a Q^*(s,a)
$$

**请记住这个"处处同时最优"的存在性结论——它是第 3 步中交换 $ \max $ 与求和次序的合法性来源。**

### 第 3 步：推导 $ Q^*(s,a) $ 的表达式

直觉先行：$ Q^*(s,a) $ 是"**先强制执行动作 $ a $，之后从下一步开始采取最优策略**"的价值。严格推导分两个方向。

**方向一（上界）**：对任意策略 $ \pi $，用第二部分的结论：

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\le \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

（逐状态放大 $ V^\pi(s') \le V^*(s') $。）对左边取 $ \max_\pi $，上界保持：

$$
Q^*(s,a) \le \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

**方向二（等号可以达到）**：方向一里的放大对每个 $ s' $ 各自用了 $ V^*(s') $，而 $ \max_\pi $ 要求**同一个策略对所有可达的 $ s' $ 同时最优**——一般而言 $ \max $ 与 $ \sum_{s'} $ 不能随意交换次序。等号成立依赖**贝尔曼最优性原理**：

> 最优策略具有这样的性质：无论初始状态与动作如何，其剩余决策序列必然构成从下一状态出发的最优策略。

取第 2 步的 $ \pi^* $（它在**所有**状态同时达到 $ V^* $），则 $ V^{\pi^*}(s') = V^*(s') $ 对全部 $ s' $ 成立，代入得

$$
Q^{\pi^*}(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

恰好达到方向一的上界。两个方向合起来：

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

如果奖励只依赖 $ (s,a) $：

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')
$$

关于"关键理解"的两点说明：

- 这一步**不是**"定义的直接体现"这么简单：把 $ \max $ 挪进对 $ s' $ 的求和里，合法性来自最优性原理与第 2 步的存在性定理——它们在有限 MDP 中成立，但都是需要证明的结论，而非记号游戏。
- 如果"之后"不按最优策略走，得到的就是某个 $ Q^\pi(s,a) $（次优值），方向一的不等号将严格成立。

### 第 4 步：得到贝尔曼最优方程

把 $ V^*(s) = \max_a Q^*(s,a) $ 代入：

$$
V^*(s) = \max_a \left[ \sum_{s'} P(s'|s,a) \left( R(s,a,s') + \gamma V^*(s') \right) \right]
$$

如果奖励只依赖 $ (s,a) $：

$$
V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
$$

这就是**贝尔曼最优方程**。

### 第 5 步：动作价值形式的贝尔曼最优方程

用 $ V^*(s') = \max_{a'} Q^*(s',a') $ 替换第 3 步结果中的 $ V^*(s') $：

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]
$$

这是 Q-learning 等算法的理论基础——其更新目标 $ r + \gamma \max_{a'} Q(s',a') $ 正是本方程的一次采样。

---

## 附录：各步骤用到的前提清单

| 前提 | 使用位置 | 作用 |
|------|----------|------|
| 有限状态/动作空间 | 三·第 2 步及全文的 max/argmax | 保证最优值与最优策略存在、可取到 |
| 环境马尔可夫性 | 二·4.3 | 未来与历史无关，条件可剥离 |
| 策略平稳且马氏 | 二·4.3 | 未来动作分布只依赖当前状态 |
| 时齐性（P、R、π 不随时间变） | 二·4.3 | 价值函数与时间下标无关 |
| 贝尔曼最优性原理 + 最优策略存在性 | 三·第 3 步 | 允许 max 与对 s' 求和交换次序 |

## 延伸阅读

- [Sutton & Barto 第 3–4 章](http://incompleteideas.net/book/the-book-2nd.html)——本推导的标准参照。
- [蘑菇书第 3–4 章](https://github.com/datawhalechina/easy-rl)——中文对照讲解。
- 路线图 [Stage 1](../README.zh-CN.md#stage-1--经典-rl-基础12-周)（本讲义的位置）与 [Stage 2](../README.zh-CN.md#stage-2--深度-rl把-ppo-吃透12-周)（贝尔曼方程在 PPO 中的用武之地）。
