#!/usr/bin/env python3
"""GridWorld 动态规划求解器：值迭代与策略迭代（Stage 1 配套可运行示例）。

与 docs/bellman-equations.zh-CN.md 配对阅读，三个算法各对应推导中的一块：
- 策略评估  = 贝尔曼期望方程的不动点迭代
- 值迭代    = 贝尔曼最优方程的不动点迭代
- 策略改进  = 对 V 做贪心（argmax），即最优方程里的 max

相对常见入门实现的几个关键差异（每条都对应一类真实 bug）：
1. 初始策略必须"能到达终点"（proper policy）：γ=1 的回合制任务中，
   到不了终点的策略其 V^π = -∞，迭代式策略评估永不收敛——最常见的死循环来源。
2. 策略评估带 max_sweeps 防护：不合适的输入得到明确报错，而不是挂死。
3. 收敛判定用"贝尔曼最优性残差" max_s |V(s) − max_a q(s,a)|，而不是
   "两种算法结果一致"：V* 唯一而 π* 不唯一（平局时不同实现可合法地
   选择不同动作），残差直接证明各自最优，更有说服力。
4. 终态在两个算法的输出中统一编码为 -1（无动作）。

依赖：仅 numpy。用法：
    python3 examples/stage1_gridworld.py                     # 4x4, γ=1.0
    python3 examples/stage1_gridworld.py --gamma 0.99 --size 5
"""

from __future__ import annotations

import argparse

import numpy as np

NO_ACTION = -1


class GridWorld:
    """方形网格 MDP：左上角出发，右下角为终点；撞墙留在原地但照付步奖励。"""

    ACTION_NAMES = ["↑", "→", "↓", "←"]
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, size: int = 4, gamma: float = 1.0, step_reward: float = -1.0):
        if size < 1:
            raise ValueError("size 必须 >= 1")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma 必须在 (0, 1] 内")
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.gamma = gamma
        self.step_reward = step_reward
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.P = self._build_model()

    # ---------- 基础 ----------
    def _to_state(self, r: int, c: int) -> int:
        return r * self.size + c

    def _to_rc(self, s: int) -> tuple[int, int]:
        return s // self.size, s % self.size

    def _is_terminal(self, s: int) -> bool:
        return self._to_rc(s) == self.goal

    def _build_model(self) -> list[list[list[tuple]]]:
        """P[s][a] = [(prob, next_s, reward, done), ...]，确定性环境各列表只有一个元素。"""
        P = [[[] for _ in range(self.n_actions)] for _ in range(self.n_states)]
        for s in range(self.n_states):
            r, c = self._to_rc(s)
            for a in range(self.n_actions):
                if self._is_terminal(s):
                    P[s][a].append((1.0, s, 0.0, True))
                    continue
                dr, dc = self.MOVES[a]
                nr, nc = r + dr, c + dc
                ns = self._to_state(nr, nc) if 0 <= nr < self.size and 0 <= nc < self.size else s
                P[s][a].append((1.0, ns, self.step_reward, self._is_terminal(ns)))
        return P

    def _q_value(self, s: int, a: int, V: np.ndarray) -> float:
        q = 0.0
        for prob, ns, reward, _ in self.P[s][a]:
            q += prob * (reward + self.gamma * V[ns])
        return q

    # ---------- 求解 ----------
    def value_iteration(self, theta: float = 1e-6, max_sweeps: int = 10_000):
        """贝尔曼最优方程的不动点迭代（原地更新，Gauss–Seidel 式）。"""
        V = np.zeros(self.n_states)
        # γ<1 时按 (1-γ)/γ 缩放阈值，保证停止时总误差有界；γ=1 的回合制任务直接用 theta
        theta_eff = theta * (1.0 - self.gamma) / self.gamma if self.gamma < 1.0 else theta
        for sweep in range(1, max_sweeps + 1):
            delta = 0.0
            for s in range(self.n_states):
                if self._is_terminal(s):
                    continue
                v = V[s]
                V[s] = max(self._q_value(s, a, V) for a in range(self.n_actions))
                delta = max(delta, abs(v - V[s]))
            if delta < theta_eff:
                return V, self._extract_policy(V), sweep
        raise RuntimeError(f"值迭代 {max_sweeps} 轮未收敛，请检查 gamma/theta 设置")

    def policy_iteration(self, theta: float = 1e-6, max_sweeps_per_eval: int = 10_000):
        """评估 + 改进交替。初始策略选保证能到达终点的"先下后右"。"""
        # 见模块 docstring 第 1 条：不可达终点的初始策略会让 γ=1 的评估发散
        policy = np.array(
            [2 if s // self.size < self.size - 1 else 1 for s in range(self.n_states)],
            dtype=int,
        )
        policy[self._to_state(*self.goal)] = NO_ACTION
        rounds = 0
        while True:
            rounds += 1
            V = self._policy_evaluation(policy, theta, max_sweeps_per_eval)
            policy, stable = self._policy_improvement(policy, V)
            if stable:
                return V, policy, rounds

    def _policy_evaluation(self, policy: np.ndarray, theta: float, max_sweeps: int):
        """贝尔曼期望方程的不动点迭代（原地更新）。"""
        V = np.zeros(self.n_states)
        for _ in range(max_sweeps):
            delta = 0.0
            for s in range(self.n_states):
                if self._is_terminal(s):
                    continue
                v = V[s]
                V[s] = self._q_value(s, policy[s], V)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                return V
        raise RuntimeError(
            "策略评估未收敛：当前策略可能无法到达终点（proper policy 问题），"
            "这在 γ=1 时意味着 V^π = -∞；请更换能到达终点的初始策略或改用 γ<1"
        )

    def _policy_improvement(self, policy: np.ndarray, V: np.ndarray):
        """对 V 贪心。1e-9 的严格改进阈值防止平局导致的状态翻转震荡。"""
        new_policy = policy.copy()
        stable = True
        for s in range(self.n_states):
            if self._is_terminal(s):
                continue
            qs = [self._q_value(s, a, V) for a in range(self.n_actions)]
            best = int(np.argmax(qs))
            if qs[best] > qs[policy[s]] + 1e-9:
                new_policy[s] = best
                stable = False
        return new_policy, stable

    def _extract_policy(self, V: np.ndarray) -> np.ndarray:
        policy = np.full(self.n_states, NO_ACTION, dtype=int)
        for s in range(self.n_states):
            if self._is_terminal(s):
                continue
            policy[s] = int(np.argmax([self._q_value(s, a, V) for a in range(self.n_actions)]))
        return policy

    def bellman_residual(self, V: np.ndarray) -> float:
        """max_s |V(s) − max_a q(s,a)|。为 0（或数值级别地小）即满足贝尔曼最优方程。"""
        return max(
            (
                abs(V[s] - max(self._q_value(s, a, V) for a in range(self.n_actions)))
                for s in range(self.n_states)
                if not self._is_terminal(s)
            ),
            default=0.0,
        )

    # ---------- 打印与模拟 ----------
    def print_values(self, V: np.ndarray, title: str) -> None:
        print(f"\n{title}:")
        border = "+" + "-------+" * self.size
        print(border)
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                if (r, c) == self.goal:
                    row += "  GOAL |"
                else:
                    row += f"{V[self._to_state(r, c)]:7.2f}|"
            print(row)
            print(border)

    def print_policy(self, policy: np.ndarray, title: str) -> None:
        print(f"\n{title}:")
        border = "+" + "-------+" * self.size
        print(border)
        for r in range(self.size):
            row = "|"
            for c in range(self.size):
                s = self._to_state(r, c)
                if (r, c) == self.goal:
                    row += "  GOAL |"
                else:
                    row += f"   {self.ACTION_NAMES[policy[s]]}   |"
            print(row)
            print(border)

    def simulate(self, policy: np.ndarray, max_steps: int = 100):
        """按（确定性）策略走一遍。取 P[s][a][0] 仅因环境确定；随机环境需按 prob 采样。"""
        s = self._to_state(*self.start)
        path = [self._to_rc(s)]
        total_reward = 0.0
        for _ in range(max_steps):
            if self._is_terminal(s):
                break
            _, ns, reward, _ = self.P[s][policy[s]][0]
            total_reward += reward
            s = ns
            path.append(self._to_rc(s))
        return path, total_reward


def main() -> None:
    parser = argparse.ArgumentParser(description="GridWorld 值迭代/策略迭代演示")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--step-reward", type=float, default=-1.0)
    args = parser.parse_args()

    env = GridWorld(size=args.size, gamma=args.gamma, step_reward=args.step_reward)
    print("=" * 55)
    print(f"{env.size}x{env.size} Gridworld MDP")
    print("=" * 55)
    print(f"起点: {env.start}    终点: {env.goal}")
    print(f"折扣因子 γ = {env.gamma}    每步奖励 = {env.step_reward}")
    print("动作: 0=↑, 1=→, 2=↓, 3=←")

    V_vi, policy_vi, sweeps_vi = env.value_iteration()
    print(f"\n--- 值迭代（{sweeps_vi} 轮扫描收敛）---")
    env.print_values(V_vi, "值迭代得到的状态价值 V(s)")
    env.print_policy(policy_vi, "值迭代得到的最优策略 π(s)")

    V_pi, policy_pi, rounds_pi = env.policy_iteration()
    print(f"\n--- 策略迭代（{rounds_pi} 轮外层迭代收敛，每轮含一次完整策略评估）---")
    env.print_values(V_pi, "策略迭代得到的状态价值 V(s)")
    env.print_policy(policy_pi, "策略迭代得到的最优策略 π(s)")

    print("\n--- 一致性与最优性检查 ---")
    print(f"两算法 V 的最大差异: {np.max(np.abs(V_vi - V_pi)):.2e}")
    for name, V in (("值迭代", V_vi), ("策略迭代", V_pi)):
        print(f"{name}的贝尔曼最优性残差 max|V−max q|: {env.bellman_residual(V):.2e}")
    # 终态已统一编码为 -1，无需掩码即可比较
    if np.array_equal(policy_vi, policy_pi):
        print("两算法策略逐状态一致。")
    else:
        diff = int(np.sum(policy_vi != policy_pi))
        print(
            f"两算法策略在 {diff} 个状态上不同——π* 不唯一（平局时裁决可不同），"
            f"但上方残差为 0 说明两者都是最优的。"
        )

    path, total_reward = env.simulate(policy_vi)
    print(f"\n从起点按最优策略走一遍:")
    print(f"路径: {path}")
    print(f"总奖励: {total_reward}    步数: {len(path) - 1}")


if __name__ == "__main__":
    main()
