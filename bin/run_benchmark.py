#!/usr/bin/env python3
"""
🚀 Entry Point: Memory System Benchmark
=========================================
Usage:
    python bin/run_benchmark.py --data-dir /path/to/data --queries 100 --systems v51,v522
"""
import sys
import os
import json
import argparse
from pathlib import Path

# 将 bin 目录加入路径以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_engine import DataLoader, TokenCounter, QueryGenerator, BenchmarkRunner
from adapters.v51_adapter import V51BaselineAdapter
from adapters.v522_adapter import V522SynapseAdapter

def main():
    parser = argparse.ArgumentParser(description="Scientific Memory System Benchmark")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to memory data (nodes/ and edges/)")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to generate")
    parser.add_argument("--hop-type", type=str, default="2hop", choices=["2hop", "3hop"], help="Query type")
    parser.add_argument("--systems", type=str, default="v51,v522", help="Comma separated list of systems to test (v51,v522)")
    args = parser.parse_args()

    print(f"📊 1. 初始化环境...")
    data = DataLoader(args.data_dir)
    stats = data.stats()
    print(f"   - 数据集: {stats['nodes']} 节点, {stats['edges']} 边")
    
    counter = TokenCounter()
    runner = BenchmarkRunner(data, counter)
    query_gen = QueryGenerator(data)
    
    # 2. 生成查询
    print(f"\n📝 2. 生成 {args.queries} 个 {args.hop_type} 查询...")
    queries = query_gen.generate_multihop_queries(n=args.queries, hop_type=args.hop_type)
    if not queries:
        print("❌ 无法生成查询，退出。")
        return
    print(f"   ✅ 成功生成 {len(queries)} 个有效查询")

    # 3. 选择系统
    systems_map = {
        "v51": V51BaselineAdapter,
        "v522": V522SynapseAdapter
    }
    
    selected_systems = [s.strip() for s in args.systems.split(",")]
    
    results = {}
    for sys_id in selected_systems:
        if sys_id in systems_map:
            adapter_class = systems_map[sys_id]
            # 实例化适配器 (传入 data_loader)
            system_instance = adapter_class(data)
            
            res = runner.run_system(system_instance.name, system_instance, queries)
            results[sys_id] = res
        else:
            print(f"⚠️ 未知系统: {sys_id}")

    # 4. 报告
    print("\n" + "="*80)
    print("📊 3. 最终测试报告")
    print("="*80)
    
    headers = ["System Name", "Recall", "Avg Tokens", "Avg Latency", "Queries"]
    print(f"{headers[0]:<45} | {headers[1]:<8} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]}")
    print("-" * 85)
    
    # Sort by recall desc
    sorted_res = sorted(results.items(), key=lambda x: x[1]['recall_pct'], reverse=True)
    
    for sid, r in sorted_res:
        print(f"{r['system']:<45} | {r['recall_pct']:>6.1f}% | {r['avg_tokens']:>8.0f} T | {r['avg_latency']:>8.1f} ms | {r['success']}/{r['total']}")

    # 对比分析
    if "v51" in results and "v522" in results:
        v1 = results["v51"]
        v2 = results["v522"]
        delta_recall = v2['recall_pct'] - v1['recall_pct']
        delta_tokens = v2['avg_tokens'] - v1['avg_tokens']
        
        print("\n💡 结论:")
        if delta_recall > 0:
            print(f"   ✅ v5.2.2 胜出! 召回率提升 {delta_recall:.1f}% (代价: +{delta_tokens:.0f} Tokens)")
        else:
            print(f"   ⚠️ v5.2.2 未胜出。召回率变化 {delta_recall:.1f}%")
            if delta_tokens > 0:
                print(f"   ⚠️ 且 Token 消耗增加了 {delta_tokens:.0f}")

    # 保存结果
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存至: {output_file}")

if __name__ == "__main__":
    main()