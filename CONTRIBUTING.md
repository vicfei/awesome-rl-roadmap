# Contributing / 贡献指南

English first, 中文摘要在后半部分。

## What we accept

This roadmap is curated with one criterion: **usefulness to an engineer on the
RL-for-LLM path**. Concretely:

- **Every entry must have a one-line "why it matters"** — what it teaches, at
  which stage, and when *not* to read it. Bare links are rejected.
- Entries belong to a **stage** (0–5). If a resource genuinely serves two
  stages, link it in both with stage-specific comments.
- Prefer primary sources: the paper itself, the official repo, the author's
  blog. Secondhand listicles don't make the cut.
- Hardware claims ("runs on one 24GB card") must come from the project's own
  docs or your own verified run. Say which.

## PR checklist

- [ ] One-line comment in **both** `README.md` and `README.zh-CN.md` (keep the
      two files structurally in sync).
- [ ] Link verified (no dead links, no paywalled mirrors when an official free
      version exists).
- [ ] Placed in the right stage; if you created a new sub-section, it fits the
      stage's goal and exit check.
- [ ] No `awesome-list padding`: we would rather have 80 sharp entries than
      800 links.

## Digest improvements

The weekly digest (`scripts/arxiv_digest.py`) is a filter, not an editor.
Improvements to keep-filter precision, tag quality, or output format are very
welcome. Keep it stdlib-only.

## 中文摘要

- 收录标准：**每条必须有一句话说明"为什么重要"**，裸链接不收；条目必须归入具体 Stage；优先一手来源（论文原文、官方仓库、作者博客）。
- 提交 PR 时请同时更新 `README.md`（英文）与 `README.zh-CN.md`（中文），保持结构一致。
- 硬件相关的说法（"单张 24GB 可跑"）需来自项目官方文档或你验证过的实测，注明出处。
- 周报脚本欢迎改进过滤精度与标签质量，保持仅依赖标准库。
