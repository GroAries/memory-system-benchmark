#!/usr/bin/env python3
"""
🚀 Entry Point for Memory System Benchmark
"""

import argparse
import json
from pathlib import Path
from benchmark_engine import (
    DataLoader, DNAv51, DNAv516, MemoryPalace, 
    Mem0Simulation, LettaSimulation, NaiveRAG, BenchmarkEvaluator
)

def main():
    parser = argparse.ArgumentParser(description="Scientific Memory System Benchmark")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to memory data directory (containing nodes/ and edges/)")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries per test type")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output results file")
    args = parser.parse_args()

    print(f"🔬 初始化数据加载: {args.data_dir}")
    data = DataLoader(args.data_dir)
    stats = data.stats()
    print(f"📊 数据集状态: {stats['nodes']} 节点, {stats['edges']} 边")

    if stats['nodes'] == 0:
        print("❌ 错误：未找到节点数据。请检查 --data-dir。")
        return

    evaluator = BenchmarkEvaluator(data)
    systems = [
        DNAv51(data, evaluator.counter),
        DNAv516(data, evaluator.counter),
        MemoryPalace(data, evaluator.counter),
        Mem0Simulation(data, evaluator.counter),
        LettaSimulation(data, evaluator.counter),
        NaiveRAG(data, evaluator.counter),
    ]

    print("\n🚀 开始测试...")
    niah_res = evaluator.test_niah(systems, n_queries=args.queries)
    multi_res = evaluator.test_multihop(systems, n_queries=args.queries)

    output = {
        "metadata": {"dataset": stats, "queries": args.queries, "token_counter": "tiktoken cl100k_base" if evaluator.counter._use_tiktoken else "estimate"},
        "niah": niah_res,
        "multihop": multi_res
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 测试完成。结果已保存至: {args.output}")
    
    # 打印简报
    print("\n📊 NIAH 简报:")
    for name, d in sorted(niah_res.items(), key=lambda x: x[1]['recall'], reverse=True):
        print(f"  {name:<20} 召回率: {d['recall_pct']:>6}  Token: {d['avg_tokens']:.0f}")
    
    print("\n📊 MultiHop 简报:")
    for name, d in sorted(multi_res.items(), key=lambda x: x[1]['2hop_recall'], reverse=True):
        print(f"  {name:<20} 2-Hop: {d['2hop_recall_pct']:>6}  3-Hop: {d['3hop_recall_pct']:>6}  Token: {d['avg_tokens']:.0f}")

if __name__ == "__main__":
    main()
