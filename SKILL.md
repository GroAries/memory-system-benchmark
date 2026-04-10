---
name: "memory-system-benchmark"
description: "Scientific Benchmark Tool for Agent Memory Systems v2.0 - Plugin-based, Fair Comparison Engine. Supports NIAH, MultiHop, and Token Efficiency testing."
version: "2.0.0"
author: "GroAries"
created: "2026-04-10"
metadata:
  category: "testing"
  tags: ["benchmark", "memory-system", "first-principles", "multi-hop", "scientific"]
  requires: ["python3", "tiktoken"]
  status: "active"
---

# 🔬 Agent Memory System Benchmark v2.0

**The Scientific Arena for Memory Systems.**
基于**第一性原理**与**控制论**重构的公平对比框架。通过插件化架构 (Plugin Architecture)，实现了真正的科学对比。

## 🌟 核心特性

1.  **绝对中立的核心 (Neutral Core)**：引擎仅负责数据加载、查询生成、Token 计量 (`tiktoken`) 和结果统计。**不包含任何具体的检索逻辑**，彻底消除了“既当裁判又当运动员”的嫌疑。
2.  **插件化适配器 (Plugin Adapters)**：
    - `v5.1`: 基于 TF-IDF + 静态 BFS 的基线实现。
    - `v5.2.2`: 基于**概念共振 (Concept Resonance)** + **贝叶斯融合 (Bayesian Fusion)** + **动态 PID 控制**的完全体实现。
    - *支持扩展*：你可以轻松编写新的 Adapter 来测试其他系统 (如 Mem0, Letta, 或向量数据库)。
3.  **全闭环多跳模拟 (Full-Loop Multi-Hop)**：不再是一次性的检索，而是模拟真实的 "Search -> Expand -> Prune -> Rescore" 迭代搜索过程。
4.  **动态资源控制 (Dynamic Budgeting)**：严格监控 Token 预算，当达到限制时自动截断，确保对比在相同资源消耗下进行。

## 🛠️ 快速使用

```bash
# 进入工具目录
cd memory-system-benchmark-skill

# 运行对比测试
# --data-dir: 记忆数据的路径 (包含 nodes/ 和 edges/)
# --queries: 查询数量
# --systems: 选择要测试的系统 (逗号分隔)
python bin/run_benchmark.py \
  --data-dir /path/to/your/data \
  --queries 100 \
  --systems v51,v522

# 仅测试 v5.2.2
python bin/run_benchmark.py \
  --data-dir /path/to/your/data \
  --queries 50 \
  --systems v522
```

## 🧩 扩展指南 (如何添加新系统?)

如果你想测试一个新的记忆系统 (比如 "MyNewMemory"):

1.  在 `bin/adapters/` 下创建 `mymemory_adapter.py`。
2.  实现 `MemorySystemAdapter` 接口：
    ```python
    from .base import MemorySystemAdapter
    class MyNewAdapter(MemorySystemAdapter):
        def retrieve(self, keywords, target_id=None, budget_tokens=5000):
            # 实现你的检索逻辑...
            # 必须返回 {"found": bool, "tokens": int, ...}
            return {"found": False, "tokens": 0}
    ```
3.  在 `bin/run_benchmark.py` 的 `systems_map` 中注册你的适配器。

## 📊 测试维度

- **2-Hop / 3-Hop Recall**: 验证多跳连通率。
- **Token Efficiency**: 找回目标所需的平均 Token 消耗。
- **Latency**: 响应时间。

---
*Powered by First Principles Thinking.*