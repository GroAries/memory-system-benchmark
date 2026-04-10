#!/usr/bin/env python3
"""
🔬 Benchmark Engine Core - Scientific Evaluation Framework
==========================================================
职责：数据加载、查询生成、Token 计量、结果统计。
原则：绝不包含具体系统的检索逻辑（由 adapters 负责），确保绝对中立。
"""

import json
import math
import random
import time
import re
import statistics
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class DataLoader:
    """统一数据加载器"""
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.nodes = {}
        self.adj = defaultdict(list) # Adjacency list
        self.edges_count = 0
        self._load()

    def _load(self):
        nodes_dir = self.data_dir / "nodes"
        if nodes_dir.exists():
            for f in nodes_dir.glob("*.json"):
                try:
                    with open(f) as file:
                        data = json.load(file)
                        nid = data.get('id', f.stem)
                        self.nodes[nid] = data
                except: pass
        
        edges_file = self.data_dir / "edges" / "edges.json"
        if edges_file.exists():
            try:
                with open(edges_file) as f:
                    edges = json.load(f)
                    self.edges_count = len(edges)
                    for e in edges.values():
                        if e['source'] in self.nodes and e['target'] in self.nodes:
                            self.adj[e['source']].append(e['target'])
                            self.adj[e['target']].append(e['source'])
            except: pass

    def stats(self) -> Dict:
        return {
            "nodes": len(self.nodes),
            "edges": self.edges_count,
            "avg_degree": sum(len(v) for v in self.adj.values()) / max(len(self.adj), 1)
        }

    def get_content(self, nid: str) -> str:
        return self.nodes.get(nid, {}).get('content', '')

    def get_neighbors(self, nid: str) -> List[str]:
        return list(set(self.adj.get(nid, [])))

class TokenCounter:
    """精确 Token 计算器"""
    def __init__(self):
        self.encoder = None
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding("cl100k_base")
            self.mode = "tiktoken"
        except ImportError:
            print("⚠️ tiktoken not found, using estimate mode.")
            self.mode = "estimate"

    def count(self, text: str) -> int:
        if not text: return 0
        if self.mode == "tiktoken":
            return len(self.encoder.encode(text))
        # 回退估算
        return int(len(text) * 0.4)

class QueryGenerator:
    """基于图结构的真实查询生成器"""
    def __init__(self, data: DataLoader):
        self.data = data
    
    def generate_multihop_queries(self, n=50, hop_type="2hop") -> List[Dict]:
        """
        生成多跳查询：
        - 2-hop: Source -> Mid -> Target
        - 返回的 Keywords 仅来自 Source，不包含 Mid 或 Target (防止作弊)
        """
        queries = []
        all_paths = []
        
        # 1. 发现所有路径
        for src in self.data.nodes:
            # 过滤无效节点
            if not self.data.get_content(src): continue
            
            for mid in self.data.get_neighbors(src):
                for dst in self.data.get_neighbors(mid):
                    if dst != src: # 简单的 2-hop 路径 (A->B->C, C!=A)
                        all_paths.append((src, dst))
                        
                        if hop_type == "3hop":
                            for fourth in self.data.get_neighbors(dst):
                                if fourth not in [src, mid, dst]:
                                    all_paths.append((src, fourth))

        if not all_paths:
            print("❌ 未找到足够的路径生成查询。")
            return []

        # 2. 抽样
        random.shuffle(all_paths)
        selected = all_paths[:n]
        
        for src, dst in selected:
            # 仅提取 Source 的关键词
            content = self.data.get_content(src)
            kws = re.findall(r'[\u4e00-\u9fff]{2,}', content)[:2]
            if kws:
                queries.append({
                    "type": hop_type,
                    "source": src,
                    "target": dst,
                    "keywords": kws
                })
        return queries

class EvaluationResult:
    """单次查询结果"""
    def __init__(self, found: bool, tokens: int, latency_ms: float, steps: int = 0, meta: Dict = None):
        self.found = found
        self.tokens = tokens
        self.latency_ms = latency_ms
        self.steps = steps
        self.meta = meta or {}

class BenchmarkRunner:
    """测试编排器"""
    def __init__(self, data: DataLoader, counter: TokenCounter):
        self.data = data
        self.counter = counter

    def run_system(self, name: str, system_class, queries: List[Dict], budget: int = 5000) -> Dict:
        print(f"\n🚀 正在测试: {name} ...")
        results = []
        
        for i, q in enumerate(queries):
            # 计时
            start = time.time()
            
            # 调用系统的 retrieve 方法
            try:
                # system 必须实现 retrieve(keywords, target_id, budget)
                res = system_class.retrieve(q['keywords'], q['target'], budget)
                latency = (time.time() - start) * 1000
                
                results.append({
                    "found": res.get("found", False),
                    "tokens": res.get("tokens", 0),
                    "latency": latency,
                    "steps": res.get("steps", 0)
                })
            except Exception as e:
                print(f"  ⚠️ 查询 {i+1} 失败: {e}")
                results.append({"found": False, "tokens": 0, "latency": 0, "steps": 0})
            
            if (i+1) % 20 == 0:
                found_count = sum(1 for r in results if r['found'])
                print(f"  进度: {i+1}/{len(queries)} (Recall: {found_count/(i+1)*100:.1f}%)")

        # 统计
        total = len(results)
        found = sum(1 for r in results if r['found'])
        return {
            "system": name,
            "recall_pct": (found / total) * 100 if total else 0,
            "avg_tokens": statistics.mean([r['tokens'] for r in results]),
            "median_tokens": statistics.median([r['tokens'] for r in results]),
            "avg_latency": statistics.mean([r['latency'] for r in results]),
            "success": found,
            "total": total
        }
