# The Bellman Equations: A Complete Derivation (Companion Notes)

> Companion derivation for [Awesome RL Roadmap](../README.md) · belongs to [Stage 1](../README.md#stage-1--classic-rl-foundations-1-2-weeks).
> Covers the full derivation of the Bellman expectation and optimality equations.
>
> **Throughout, we assume a finite MDP** (finite state and action spaces): this
> keeps every $\max$and$\arg\max$ below well-defined. For continuous action
> spaces, replace $\max$with$\sup$ and mind measurability and attainability.

## 1. First define the basic objects

### 1.1 Markov Decision Process (MDP)

An MDP is the five-tuple $(S, A, P, R, \gamma)$:

- $S$: the state space.
- $A$: the action space.
- $P(s'|s,a)$: probability of transitioning to $s'$after taking action$a$in state$s$.
- $R(s,a,s')$: immediate reward for taking $a$in$s$and landing in$s'$. Sometimes shortened to $R(s,a)$.
- $\gamma \in [0,1]$: the discount factor.

### 1.2 Policy

A policy $\pi(a|s)$is the probability of choosing action$a$in state$s$. Throughout, $\pi$ is a **stationary Markov policy**: the action distribution depends only on the current state — not on time or history.

### 1.3 Return

The discounted return from time $t$ is

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

with the key recursive form

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

**Why $\gamma$ at all?**

- Mathematically: for non-terminating tasks with strictly positive rewards, the undiscounted series diverges; $\gamma < 1$ guarantees convergence.
- Intuitively: future rewards are worth less than present ones; $\gamma$ encodes that preference.
- The recursion doesn't *add* a $\gamma$ — it extracts the discounting structure already present in the definition.

### 1.4 State-value function

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right]
$$

### 1.5 Action-value function

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right]
$$

---

## 2. Deriving the Bellman expectation equation

### Step 1: split the return, straight from the definition

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right]
$$

### Step 2: linearity of expectation

$$
V^\pi(s) = \underbrace{\mathbb{E}_\pi \left[ R_{t+1} \mid S_t = s \right]}_{\text{expected immediate reward}} + \gamma \underbrace{\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]}_{\text{expected future return}}
$$

### Step 3: the expected immediate reward

In state $s$, $A_t$is drawn from$\pi(a|s)$, then the environment transitions and pays $R(s,a,s')$. Law of total expectation, first over actions, then over next states:

$$
\mathbb{E}_\pi \left[ R_{t+1} \mid S_t = s \right]
=
\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')
$$

If the reward depends only on $(s,a)$, this shortens to $\sum_a \pi(a|s) R(s,a)$.

### Step 4: the expected future return (the step most treatments blur)

Three expansions.

#### 4.1 Expand over $A_t$

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]
=
\sum_a \underbrace{P(A_t = a \mid S_t = s)}_{\pi(a|s)} \cdot
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
$$

#### 4.2 Expand over $S_{t+1}$

For a fixed $a$, the environment transitions to $s'$with probability$P(s'|s,a)$; law of total expectation once more:

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
=
\sum_{s'} P(s'|s,a) \cdot
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \right]
$$

#### 4.3 Simplify by the Markov property (note: this actually uses three premises)

To claim

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a, S_{t+1} = s' \right]
=
\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right]
$$

the environment's Markov property alone is not enough — three premises work together:

1. **Markov environment**: given $S_{t+1}=s'$, future transitions and rewards are independent of the history $(S_t, A_t)$;
2. **Stationary Markov policy**: from $t+1$onward, actions are drawn from$\pi$based only on the then-current state, also independent of history — so conditioning on the extra$(S_t=s, A_t=a)$ does not change the distribution of future actions;
3. **Time homogeneity**: $P$, $R$, and $\pi$ do not change over time, hence

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] = V^\pi(s')
$$

the right-hand side being exactly $V^\pi$evaluated at time$t+1$— time homogeneity is what makes it the *same* function, with no need for a$V^\pi_{t+1}$.

#### 4.4 Combine

Substituting 4.3 into 4.2 and back into 4.1:

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s \right]
=
\sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')
$$

### Step 5: assemble the Bellman expectation equation

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')
+ \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) V^\pi(s')
$$

Merge the sums:

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

and if the reward depends only on $(s,a)$:

$$
V^\pi(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]
$$

This is the **Bellman expectation equation**.

### Step 6: the action-value form

From the definition of $Q^\pi$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s, A_t = a \right]
$$

The immediate-reward term is $\sum_{s'} P(s'|s,a) R(s,a,s')$; expanding the future term over $s'$ (same argument as 4.2–4.3):

$$
\mathbb{E}_\pi \left[ G_{t+1} \mid S_t = s, A_t = a \right]
=
\sum_{s'} P(s'|s,a) V^\pi(s')
$$

so

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

Together with $V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$, the two equations substitute into each other.

---

## 3. Deriving the Bellman optimality equation

### Step 1: define the optimal value functions

$$
V^*(s) = \max_\pi V^\pi(s)
\qquad
Q^*(s,a) = \max_\pi Q^\pi(s,a)
$$

### Step 2: existence and greedy property of an optimal policy

In a finite MDP one can prove: **there exists a deterministic optimal policy $\pi^*$that attains$V^*$at every state simultaneously** (proof sketch: the Bellman optimality operator is a$\gamma$-contraction, so $V^*$exists and is unique; any policy greedy w.r.t.$V^*$attains it). It picks, in each state, an action maximizing$Q^*(s,a)$:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
\qquad\text{hence}\qquad
V^*(s) = \max_a Q^*(s,a)
$$

**Remember this "optimal everywhere at once" existence result — it is what licenses swapping $\max$ and summation in Step 3.**

### Step 3: deriving the expression for $Q^*(s,a)$

Intuition first: $Q^*(s,a)$is the value of "**force action$a$ first, then act optimally from the next step on**". The rigorous derivation runs in two directions.

**Direction 1 (upper bound).** For any policy $\pi$, reusing Part 2:

$$
Q^\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\le \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

(replacing $V^\pi(s')$by$V^*(s')$state by state). Taking$\max_\pi$ on the left preserves the bound:

$$
Q^*(s,a) \le \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

**Direction 2 (the bound is attained).** Direction 1 maximized each $s'$with its *own* best value, whereas$\max_\pi$demands a **single policy that is optimal at all reachable$s'$simultaneously** — in general,$\max$and$\sum_{s'}$ cannot be interchanged freely. Equality rests on **Bellman's principle of optimality**:

> An optimal policy has the property that whatever the initial state and action, the remaining decisions must constitute an optimal policy from the state at time $t+1$.

Take the $\pi^*$from Step 2 (optimal at *every* state, so$V^{\pi^*}(s') = V^*(s')$for all$s'$); substituting,

$$
Q^{\pi^*}(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

which attains the bound from Direction 1. Both directions together:

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

and if the reward depends only on $(s,a)$:

$$
Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')
$$

Two remarks on the "key insight":

- This step is **not** merely "unfolding the definition": moving $\max$inside the sum over$s'$ is licensed by the principle of optimality plus the existence theorem of Step 2 — both true in finite MDPs, but both are theorems to be proven, not notational games.
- If the agent does *not* act optimally afterwards, the result is some $Q^\pi(s,a)$ (a suboptimal value) and the inequality in Direction 1 is strict.

### Step 4: the Bellman optimality equation

Substituting $V^*(s) = \max_a Q^*(s,a)$:

$$
V^*(s) = \max_a \left[ \sum_{s'} P(s'|s,a) \left( R(s,a,s') + \gamma V^*(s') \right) \right]
$$

or, with reward depending only on $(s,a)$:

$$
V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s') \right]
$$

This is the **Bellman optimality equation**.

### Step 5: the action-value form

Replace $V^*(s')$in Step 3's result by$V^*(s') = \max_{a'} Q^*(s',a')$:

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]
$$

This is the theoretical foundation of Q-learning and its relatives — the update target $r + \gamma \max_{a'} Q(s',a')$ is exactly one sampled instance of this equation.

---

## Appendix: which premises are used where

| Premise | Used in | Role |
|---------|---------|------|
| Finite state/action spaces | Part 3 Step 2 and every max/argmax | Optimal values and policies exist and are attained |
| Markov environment | Part 2 §4.3 | Future independent of history; conditioning can be dropped |
| Stationary Markov policy | Part 2 §4.3 | Future action distribution depends only on current state |
| Time homogeneity (P, R, π fixed) | Part 2 §4.3 | Value function carries no time index |
| Principle of optimality + existence of an everywhere-optimal policy | Part 3 Step 3 | Licenses swapping max and the sum over s' |

## Further reading

- [Companion runnable example: GridWorld value/policy iteration](../examples/stage1_gridworld.py) — this derivation turned into code (evaluation = iterating the expectation equation, value iteration = iterating the optimality equation, improvement = the max), with a proper-policy guard and Bellman-residual checks.
- [Sutton & Barto, ch. 3–4](http://incompleteideas.net/book/the-book-2nd.html) — the canonical reference for this derivation.
- [The Mushook, ch. 3–4](https://github.com/datawhalechina/easy-rl) (Chinese) — a companion walk-through.
- Roadmap [Stage 1](../README.md#stage-1--classic-rl-foundations-1-2-weeks) (where these notes live) and [Stage 2](../README.md#stage-2--deep-rl-master-ppo-1-2-weeks) (where the Bellman equations meet PPO).
