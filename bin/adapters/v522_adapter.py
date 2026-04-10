#!/usr/bin/env python3
"""
v5.2.2 Synapse Adapter: Concept Resonance + Bayesian Fusion + Dynamic Control
Logic:
1. Dual-Channel Retrieval: TF-IDF + Hebbian Resonance.
2. Bayesian Fusion: Combine scores.
3. Graph Walk: Use resonance scores to guide expansion.
"""
import sys
import os
import re
import time
import json
from collections import defaultdict
from .base import MemorySystemAdapter

# 尝试自动发现模块路径
def discover_v522_modules():
    """尝试找到本地的 agent-memory-dna-v5-1-6/bin 目录"""
    # 常见路径
    paths_to_try = [
        os.path.join(os.path.dirname(__file__), "../../../skills/agent-memory-dna-v5-1-6/bin"),
        os.path.join(os.path.dirname(__file__), "../../../../agent-memory-dna-v5-1-6/bin"),
    ]
    
    for p in paths_to_try:
        abs_p = os.path.abspath(p)
        if os.path.isdir(abs_p):
            if os.path.exists(os.path.join(abs_p, "occupancy_retriever.py")):
                sys.path.insert(0, abs_p)
                try:
                    from occupancy_retriever import OccupancyRetriever
                    from concept_resonator import ConceptResonator
                    from bayesian_fuser import BayesianSeedFuser
                    return True, OccupancyRetriever, ConceptResonator, BayesianSeedFuser
                except ImportError:
                    continue
    return False, None, None, None

class V522SynapseAdapter(MemorySystemAdapter):
    def __init__(self, data_loader):
        self.data = data_loader
        self.name = "v5.2.2 Synapse (Resonance + Fusion)"
        
        # 加载外部模块
        loaded, OccRet, ConRes, BayFus = discover_v522_modules()
        
        if loaded:
            self.OccupancyRetriever = OccRet
            self.ConceptResonator = ConRes
            self.BayesianSeedFuser = BayFus
            
            # 初始化实例
            # 将 DataLoader 的数据转换为 dict 格式供模块使用
            nodes_dict = data_loader.nodes
            adj_dict = dict(data_loader.adj)
            
            self.retriever = self.OccupancyRetriever(nodes_dict, adj_dict)
            self.resonator = self.ConceptResonator(nodes_dict)
            self.fuser = self.BayesianSeedFuser(alpha=0.4)
            self.is_ready = True
        else:
            self.is_ready = False
            print("⚠️ v5.2.2 模块未找到，将使用降级模式 (TF-IDF + Static Resonance Simulation)")

    def retrieve(self, keywords, target_id=None, budget_tokens=5000):
        start_time = time.time()
        
        if not self.is_ready:
            # 降级模式 (Fallback)
            return self._fallback_retrieve(keywords, target_id, budget_tokens, start_time)

        total_tokens = 0
        visited = set()
        
        # --- Step 1: 双通道检索 ---
        # Channel A: TF-IDF
        tfidf_scores = dict(self.retriever.query_occupancy(keywords, top_k=50))
        confidence_a = list(tfidf_scores.values())[0] if tfidf_scores else 0
        
        # Channel B: Resonance (Gated)
        resonance_triggered = False
        resonance_scores = {}
        
        # 门控阈值 (基于 v5.1 的性能基线 ~35.0)
        if confidence_a < 35.0:
            resonance_scores = self.resonator.query_resonance(keywords, self.data.nodes)
            resonance_triggered = True
            
        # Fusion
        if resonance_triggered and resonance_scores:
            fused = self.fuser.fuse(tfidf_scores, resonance_scores, mode='arithmetic')
            seeds = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        else:
            seeds = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
        # Init Frontier
        frontier = [nid for nid, _ in seeds[:5]]
        visited.update(frontier)
        
        # 估算 Token
        content_cost = int(len("".join([self.data.get_content(n) for n in frontier])) * 0.4)
        total_tokens += content_cost
        
        steps = 0
        max_steps = 4
        
        # --- Step 2: 闭环扩展 (Graph Walk) ---
        while total_tokens < budget_tokens and steps < max_steps:
            if target_id and target_id in visited:
                break
            
            # 扩展邻居
            candidates = []
            for nid in frontier:
                for neighbor in self.data.get_neighbors(nid):
                    if neighbor not in visited:
                        # 评分：优先使用共振分 (如果存在)
                        score = resonance_scores.get(neighbor, 0) if resonance_triggered else 0
                        # 如果共振分没有，退化为 TF-IDF
                        if score == 0: score = tfidf_scores.get(neighbor, 0)
                        candidates.append((neighbor, score))
                        visited.add(neighbor)
            
            if not candidates: break
            
            # 排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 动态宽度控制 (PID 模拟)
            width = 15 if total_tokens < budget_tokens * 0.5 else 8
            selected = [nid for nid, _ in candidates[:width]]
            
            if not selected: break
            
            # Budget Check
            content = "".join([self.data.get_content(n) for n in selected])
            cost = int(len(content) * 0.4)
            
            if total_tokens + cost > budget_tokens:
                # 缩减直到满足预算
                for k in range(1, len(selected)):
                    sub_content = "".join([self.data.get_content(n) for n in selected[:k]])
                    sub_cost = int(len(sub_content) * 0.4)
                    if total_tokens + sub_cost <= budget_tokens:
                        selected = selected[:k]
                        cost = sub_cost
                        break
                else:
                    break
            
            total_tokens += cost
            frontier = selected
            steps += 1
            
        return {
            "found": (target_id in visited) if target_id else False,
            "tokens": total_tokens,
            "latency": (time.time() - start_time) * 1000,
            "steps": steps,
            "meta": {"resonance_used": resonance_triggered}
        }

    def _fallback_retrieve(self, keywords, target_id, budget_tokens, start_time):
        """当模块加载失败时的简化实现，保证测试不中断"""
        # 简单 TF-IDF + BFS
        # (复用 v51 逻辑的简化版)
        from collections import defaultdict
        import re
        inv = defaultdict(list)
        for nid, node in self.data.nodes.items():
            words = set(re.findall(r'[\u4e00-\u9fff]{2,}', node.get('content', '').lower()))
            for w in words: inv[w].append(nid)
            
        seeds = set()
        for kw in keywords:
            for w, nids in inv.items():
                if kw.lower() in w: seeds.update(nids)
        
        visited = list(seeds)[:10]
        return {
            "found": (target_id in visited) if target_id else False,
            "tokens": 1000,
            "latency": (time.time() - start_time) * 1000,
            "steps": 1,
            "meta": {"fallback": True}
        }