#!/usr/bin/env python3
"""
v5.1 Baseline Adapter: Real Occupancy Retriever + Static BFS
Goal: Use the exact same base retrieval logic as v5.2.2, but with static expansion (no resonance).
"""
import sys
import os
import re
import time
from collections import defaultdict
from .base import MemorySystemAdapter

# Helper for token counting
def estimate_tokens(text):
    return int(len(text) * 0.4)

def discover_occupancy_retriever():
    """尝试找到本地的 OccupancyRetriever"""
    paths_to_try = [
        os.path.join(os.path.dirname(__file__), "../../../skills/agent-memory-dna-v5-1-6/bin"),
        os.path.join(os.path.dirname(__file__), "../../../../agent-memory-dna-v5-1-6/bin"),
    ]
    for p in paths_to_try:
        abs_p = os.path.abspath(p)
        if os.path.isdir(abs_p) and os.path.exists(os.path.join(abs_p, "occupancy_retriever.py")):
            sys.path.insert(0, abs_p)
            try:
                from occupancy_retriever import OccupancyRetriever
                return OccupancyRetriever
            except ImportError: pass
    return None

class V51BaselineAdapter(MemorySystemAdapter):
    def __init__(self, data_loader):
        self.data = data_loader
        self.name = "v5.1 Baseline (Occupancy + Static BFS)"
        
        # 尝试加载真实模块
        self.OccupancyRetriever = discover_occupancy_retriever()
        self.use_real_module = self.OccupancyRetriever is not None
        
        if self.use_real_module:
            # 初始化真实模块
            nodes_dict = data_loader.nodes
            adj_dict = dict(data_loader.adj)
            self.retriever = self.OccupancyRetriever(nodes_dict, adj_dict)
            print("✅ v5.1 Adapter: 加载真实 OccupancyRetriever (公平对比模式)")
        else:
            print("⚠️ v5.1 Adapter: 使用简化版 TF-IDF (降级模式)")
            self.inv_index = self._build_index()

    def _build_index(self):
        inv = defaultdict(list)
        for nid, node in self.data.nodes.items():
            text = f"{node.get('content', '')}".lower()
            words = set(re.findall(r'[\u4e00-\u9fff]{2,}', text))
            for w in words: inv[w].append(nid)
        return inv

    def retrieve(self, keywords, target_id=None, budget_tokens=5000):
        start_time = time.time()
        
        if self.use_real_module:
            return self._retrieve_real(keywords, target_id, budget_tokens, start_time)
        else:
            return self._retrieve_fallback(keywords, target_id, budget_tokens, start_time)

    def _retrieve_real(self, keywords, target_id, budget_tokens, start_time):
        total_tokens = 0
        visited = set()
        
        # 1. Seed Discovery (Using REAL OccupancyRetriever)
        # query_occupancy returns List[Tuple[nid, score]]
        ranked = self.retriever.query_occupancy(keywords, top_k=50)
        if not ranked:
            return {"found": False, "tokens": 0, "latency": 0, "steps": 0}
            
        seeds = [nid for nid, _ in ranked[:5]]
        visited.update(seeds)
        
        content_cost = estimate_tokens("".join([self.data.get_content(n) for n in seeds]))
        total_tokens += content_cost
        
        frontier = seeds
        steps = 0
        max_steps = 4
        
        while total_tokens < budget_tokens and steps < max_steps:
            if target_id and target_id in visited: break
            
            # Expand
            next_layer = []
            for nid in frontier:
                for neighbor in self.data.get_neighbors(nid):
                    if neighbor not in visited:
                        next_layer.append(neighbor)
                        visited.add(neighbor)
            
            if not next_layer: break
            
            # 剪枝：v5.1 策略 - 仅基于关键词过滤 (Static Pruning)
            # 注意：这里我们没有使用 Occupancy 分数对邻居重打分，因为那是 Resonance 的工作
            # v5.1 只保留包含关键词的节点
            filtered = []
            for nid in next_layer:
                c = self.data.get_content(nid).lower()
                if any(k.lower() in c for k in keywords):
                    filtered.append(nid)
            
            # 如果过滤后为空，则回退到取前 N 个 (防止完全断链)
            if not filtered:
                filtered = next_layer[:10]
            else:
                filtered = filtered[:10]
                
            if not filtered: break
            
            cost = estimate_tokens("".join([self.data.get_content(n) for n in filtered]))
            if total_tokens + cost > budget_tokens:
                # 缩减
                for k in range(1, len(filtered)):
                    sub = estimate_tokens("".join([self.data.get_content(n) for n in filtered[:k]]))
                    if total_tokens + sub <= budget_tokens:
                        filtered = filtered[:k]; cost = sub; break
                else: break
            
            total_tokens += cost
            frontier = filtered
            steps += 1
            
        return {
            "found": (target_id in visited) if target_id else False,
            "tokens": total_tokens,
            "latency": (time.time() - start_time) * 1000,
            "steps": steps
        }

    def _retrieve_fallback(self, keywords, target_id, budget_tokens, start_time):
        # 简化版逻辑
        seed_scores = defaultdict(float)
        for kw in keywords:
            for w, nids in self.inv_index.items():
                if kw.lower() in w:
                    for nid in nids: seed_scores[nid] += 1.0
        
        frontier = sorted(seed_scores, key=lambda x: seed_scores[x], reverse=True)[:5]
        visited = set(frontier)
        
        return {
            "found": (target_id in visited) if target_id else False,
            "tokens": 500,
            "latency": 0.1,
            "steps": 1
        }