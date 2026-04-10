# 🔬 Agent Memory System Benchmark v2.0

**The Scientific Arena for Memory Systems.**

本项目是一个基于**第一性原理 (First Principles)** 与**控制论 (Cybernetics)** 构建的公平对比框架。旨在通过科学的量化指标（如 Token 效率、多跳召回率、延迟），评估不同 Agent 记忆系统在真实场景下的表现。

---

## 🌟 核心特性

1.  **绝对中立 (Neutral Core)**：
    评测引擎 (`benchmark_engine.py`) **仅负责**数据加载、查询生成和 Token 计量。它**不包含**任何具体的检索逻辑。
    *这确保了评测框架绝不会“既当裁判又当运动员”。*

2.  **插件化架构 (Plugin Architecture)**：
    每个记忆系统都是一个独立的适配器 (Adapter)。你可以轻松插入 v5.1, v5.2.2, 甚至第三方的向量数据库进行对比。

3.  **全闭环多跳模拟 (Full-Loop Multi-Hop)**：
    不仅仅是一次简单的关键词匹配。测试模拟了真实的 **Search -> Expand -> Prune -> Rescore** 迭代搜索循环。

4.  **真实模块集成 (Real Module Integration)**：
    支持动态发现并加载本地实际部署的记忆系统模块（如 `OccupancyRetriever`, `ConceptResonator`），确保测试结果反映真实运行状况。

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install tiktoken
```

### 2. 准备数据
确保你的数据目录结构如下：
```text
/path/to/data/
├── nodes/
│   ├── node1.json
│   └── ...
└── edges/
    └── edges.json
```

### 3. 运行基准测试
```bash
python bin/run_benchmark.py \
  --data-dir /path/to/your/data \
  --queries 50 \
  --systems v51,v522
```

**输出示例：**
```text
================================================================================
📊 3. 最终测试报告
================================================================================
System Name                                   | Recall   | Avg Tokens | Avg Latency
-------------------------------------------------------------------------------------
v5.2.2 Synapse (Resonance + Fusion)           |   65.0%  |     2220 T |      2.7 ms
v5.1 Baseline (Occupancy + Static BFS)        |   55.0%  |      983 T |      2.0 ms

💡 结论:
   ✅ v5.2.2 胜出! 召回率提升 10.0% (代价: +1237 Tokens)
```

---

## 🧩 如何添加新系统 (Add Your Own System)

本项目的设计允许任何人扩展对比范围。

1.  **创建适配器**：在 `bin/adapters/` 下新建文件（如 `my_system_adapter.py`）。
2.  **继承基类**：
    ```python
    from .base import MemorySystemAdapter

    class MySystemAdapter(MemorySystemAdapter):
        def __init__(self, data_loader):
            self.data = data_loader
            self.name = "My Super System"
            # 初始化你的索引或模型

        def retrieve(self, keywords, target_id=None, budget_tokens=5000):
            # 实现检索逻辑...
            # 注意：target_id 仅供内部验证，请勿直接用于返回结果（作弊）
            
            # 必须返回字典:
            return {
                "found": True/False, 
                "tokens": int, 
                "latency": float
            }
    ```
3.  **注册**：在 `bin/run_benchmark.py` 的 `systems_map` 中添加你的适配器映射。

---

## 📚 包含的系统说明

| 系统 ID | 名称 | 架构简述 |
| :--- | :--- | :--- |
| **v51** | **v5.1 Baseline** | 确定性 TF-IDF + 静态广度优先搜索 (BFS)。依靠关键词匹配，低 Token 消耗。 |
| **v522** | **v5.2.2 Synapse** | 概念共振 + 贝叶斯融合 + PID 动态控制。引入 Hebbian 联想机制，擅长处理语义鸿沟。 |

---

**Maintainer**: Jarvis / Oracle  
**Version**: 2.0.0
