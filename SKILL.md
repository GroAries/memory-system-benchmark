---
name: "memory-system-benchmark"
description: "Scientific Benchmark Tool for Agent Memory Systems - Based on First Principles. Supports NIAH, MultiHop, and Token Efficiency testing for DNA, Memory Palace, Mem0, etc."
version: "1.0.0"
author: "GroAries"
created: "2026-04-10"
metadata:
  category: "testing"
  tags: ["benchmark", "memory-system", "first-principles", "NIAH", "MultiHop", "DNA-v5"]
  requires: { "python": ["tiktoken"] }
  status: "active"
---

# 🔬 Agent Memory System Benchmark (Scientific Edition)

基于**第一性原理**重构的公平对比框架，用于科学评估各类 Agent 记忆系统的性能。

## 核心特性
1. **公平对比**：所有系统在同一数据集上运行，使用同一查询集。
2. **真实实现**：不硬编码结果，每个系统实现真实的检索逻辑。
3. **精确计量**：使用 `tiktoken` (cl100k_base) 精确计算 Token 消耗。
4. **统计可靠**：100+ 次抽样，报告均值 ± 标准差。

## 支持系统
- **DNA v5.1**: 确定性记忆 + 世界模型 (World Model)
- **DNA v5.1.6**: 双重双层脑 + 控制论自进化 (Dual-Brain)
- **记忆宫殿**: 全量加载方案 (Memory Palace)
- **Mem0**: 向量 + 图谱混合 (SOTA Simulation)
- **Letta**: 上下文分块管理 (MemGPT Simulation)
- **Naive RAG**: 纯向量检索

## 测试维度
- **NIAH (Needle In A Haystack)**: 长文本精确召回率
- **MultiHop (2-Hop / 3-Hop)**: 多跳逻辑推理连通率
- **Token Efficiency**: 检索过程中的 Token 消耗效率
- **Latency**: 检索响应延迟

## 快速开始

```bash
# 1. 安装依赖
pip install tiktoken

# 2. 准备数据
# 将记忆节点 JSON 放置在 data/nodes/ 目录下
# 将图谱边 JSON 放置在 data/edges/edges.json

# 3. 运行测试
python run_benchmark.py --data-dir ./data --queries 100

# 4. 生成报告
python report_generator.py --results benchmark_results.json
```

## 科学方法论
- **控制变量**：同数据集、同查询集、同 Token 计算器。
- **大样本**：默认 100 次随机查询，消除偶然性。
- **透明**：所有系统逻辑开源可查，拒绝黑盒。

---
*Designed with First Principles Thinking & Cybernetics.*
