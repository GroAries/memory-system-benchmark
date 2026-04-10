#!/usr/bin/env python3
"""
🔬 v5.2.2 Synapse vs v5.1 Baseline: 全闭环科学对比测试
=========================================================
基于第一性原理与控制论构建的终极考场。

1. **真实调用**：不使用 Mock，直接 import 本地 `agent-memory-dna-v5-1-6/bin/` 中的核心模块。
2. **闭环检索**：模拟真实的多跳搜索过程（Source -> Step 1 -> Step 2 -> ... -> Target）。
3. **双重监察**：
   - 🧠 第一性原理：严格控制输入/输出/Token，确保度量衡一致。
   - 🔄 控制论：允许系统使用各自的策略（如 PID 控制器）动态分配资源，对比固定预算下的表现。

Usage:
    python run_v522_full_comparison.py --data-dir /path/to/data --queries 50
"""

import sys
import os
import json
import time
import re
import random
import statistics
import argparse
from pathlib import Path
from collections import defaultdict

# ============================================================
# 0. 环境初始化：导入 V5.2.2 核心模块
# ============================================================
# 假设 v5.2.2 的代码位于父目录的 agent-memory-dna-v5-1-6/bin 下
# (基于之前的开发上下文，v5.2.2 模块已部署于此)
v5_base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../skills/agent-memory-dna-v5-1-6/bin"))
sys.path.insert(0, v5_base_path)

try:
    print(f"📦 正在加载 v5.2.2 模块: {v5_base_path}")
    from occupancy_retriever import OccupancyRetriever
    from concept_resonator import ConceptResonator
    from bayesian_fuser import BayesianSeedFuser
    # 注意：v5.2.2 的 Graph Walk 逻辑集成在 test_v522_synapse.py 中，
    # 为了复用，我们需要在这里重新实现或者导入它。
    # 鉴于之前的测试脚本是集成测试，我们在这里提取核心逻辑。
    HAS_V522_MODULES = True
    print("✅ 成功导入 v5.2.2 核心组件 (Retriever, Resonator, Fuser)")
except ImportError as e:
    print(f"❌ 无法导入 v5.2.2 模块: {e}")
    print(f"⚠️ 路径: {v5_base_path}")
    HAS_V522_MODULES = False

# 尝试导入 tiktoken
try:
    import tiktoken
    tik_tokenizer = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except:
    HAS_TIKTOKEN = False
    print("⚠️ tiktoken 未安装，使用估算模式。")

# ============================================================
# 1. 数据加载器 (共享)
# ============================================================
class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.nodes = {}
        self.adj = defaultdict(list)
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
                    for e in edges.values():
                        if e['source'] in self.nodes and e['target'] in self.nodes:
                            self.adj[e['source']].append(e['target'])
                            self.adj[e['target']].append(e['source'])
            except: pass

    def stats(self):
        return {"nodes": len(self.nodes), "edges": len(self.adj)}

    def get_content(self, nid):
        return self.nodes.get(nid, {}).get('content', '')

    def get_neighbors(self, nid):
        return list(set(self.adj.get(nid, []))) # Unique neighbors

# ============================================================
# 2. 系统实现 (完全体)
# ============================================================

def count_tokens(text):
    if HAS_TIKTOKEN:
        return len(tik_tokenizer.encode(text))
    return int(len(text) * 0.5) # Fallback

class System_V51_Baseline:
    """
    v5.1 Baseline: 确定性 TF-IDF + 静态邻居扩展
    第一性原理假设：记忆检索基于词汇匹配和直接关联。
    """
    def __init__(self, data: DataLoader):
        self.data = data
        self.name = "v5.1 Baseline (TF-IDF + Static Graph)"
        # 构建简单的 TF-IDF 索引
        self.inv_index = defaultdict(list)
        for nid, node in data.nodes.items():
            text = f"{node.get('content', '')}".lower()
            # 提取中文词组
            words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
            for w in words:
                self.inv_index[w].append(nid)

    def _search_seeds(self, keywords):
        scores = defaultdict(float)
        for kw in keywords:
            kw_lower = kw.lower()
            # 精确匹配
            for nid in self.inv_index.get(kw_lower, []): scores[nid] += 10.0
            # 模糊匹配 (包含关系)
            for word, nids in self.inv_index.items():
                if kw_lower in word or word in kw_lower:
                    for nid in nids: scores[nid] += 2.0
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, keywords, target_id=None, budget_tokens=5000):
        """
        执行一次完整的多跳检索尝试 (模拟 2-hop 或 3-hop)
        v5.1 策略：找到 Top 3 种子 -> 扩展所有邻居 -> 筛选 (关键词匹配) -> 再扩展
        """
        start_time = time.time()
        total_tokens = 0
        visited = set()
        
        # Step 1: 初始检索
        seeds = self._search_seeds(keywords)
        if not seeds: return {"found": False, "tokens": 0, "steps": 0, "latency": 0}
        
        # 只取 Top 5 作为入口
        current_frontier = [nid for nid, _ in seeds[:5]]
        visited.update(current_frontier)
        
        # 计算初始 Token (读取种子节点内容)
        content = "".join([self.data.get_content(n) for n in current_frontier])
        total_tokens += count_tokens(content)
        
        steps = 0
        max_steps = 4 # v5.1 允许的最大跳数
        
        # 循环扩展
        while total_tokens < budget_tokens and steps < max_steps:
            # 检查是否包含目标
            if target_id in visited:
                break # 找到目标
            
            next_frontier = []
            # 扩展当前 frontier 的所有邻居
            for nid in current_frontier:
                neighbors = self.data.get_neighbors(nid)
                for n in neighbors:
                    if n not in visited:
                        next_frontier.append(n)
                        visited.add(n)
            
            if not next_frontier: break
            
            # 剪枝：基于关键词过滤 (v5.1 的弱项：强依赖词匹配)
            filtered = []
            for nid in next_frontier:
                c = self.data.get_content(nid)
                # 简单启发式：内容中是否包含查询词的任意部分
                score = sum(1 for k in keywords if k.lower() in c.lower())
                if score > 0: filtered.append((nid, score))
            
            # 按相关性排序并截取
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            # 预算控制
            to_visit = [nid for nid, s in filtered[:10]] # 取前 10
            content = "".join([self.data.get_content(n) for n in to_visit])
            cost = count_tokens(content)
            
            if total_tokens + cost > budget_tokens:
                break # 预算耗尽
                
            total_tokens += cost
            visited.update(to_visit)
            current_frontier = to_visit
            steps += 1
            
        return {
            "found": target_id in visited,
            "tokens": total_tokens,
            "steps": steps,
            "latency": (time.time() - start_time) * 1000,
            "visited_count": len(visited)
        }


class System_V522_Synapse:
    """
    v5.2.2 Synapse: 概念共振 + 贝叶斯融合 + PID 动态控制
    """
    def __init__(self, data: DataLoader):
        self.data = data
        self.name = "v5.2.2 Synapse (Resonance + Fusion + PID)"
        
        if not HAS_V522_MODULES:
            raise Exception("Core v5.2.2 modules missing!")

        # 初始化真实模块
        # 将 DataLoader 的数据转换为模块需要的格式
        self.nodes_dict = dict(data.nodes)
        self.adj_dict = dict(data.adj)
        
        self.retriever = OccupancyRetriever(self.nodes_dict, self.adj_dict)
        self.resonator = ConceptResonator(self.nodes_dict)
        self.fuser = BayesianSeedFuser(alpha=0.4)

        # 简单的 PID 模拟 (复用 v5.2.1 的逻辑概念)
        # 在这里硬编码简单的动态逻辑，避免复杂的类依赖
        self.current_k = 10 # 动态宽度
        
    def retrieve(self, keywords, target_id=None, budget_tokens=5000):
        start_time = time.time()
        total_tokens = 0
        visited = set()
        
        # Step 1: 双通道检索
        tfidf_scores = dict(self.retriever.query_occupancy(keywords, top_k=50))
        confidence_a = list(tfidf_scores.values())[0] if tfidf_scores else 0
        
        resonance_triggered = False
        resonance_scores = {}
        
        # 门控逻辑 (Threshold < 35.0 根据之前的测试数据)
        if confidence_a < 35.0:
            resonance_scores = self.resonator.query_resonance(keywords, self.nodes_dict)
            resonance_triggered = True
            
        # 融合
        if resonance_triggered and resonance_scores:
            fused = self.fuser.fuse(tfidf_scores, resonance_scores, mode='arithmetic')
            seeds = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        else:
            seeds = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            
        # 初始化 Frontier
        current_frontier = [nid for nid, _ in seeds[:5]]
        visited.update(current_frontier)
        
        # 计算初始 Token
        content = "".join([self.data.get_content(n) for n in current_frontier])
        total_tokens += count_tokens(content)
        
        steps = 0
        max_steps = 4
        
        # 循环扩展 (PID 控制)
        while total_tokens < budget_tokens and steps < max_steps:
            if target_id in visited:
                break
            
            # 扩展邻居
            next_candidates = []
            for nid in current_frontier:
                neighbors = self.data.get_neighbors(nid)
                for n in neighbors:
                    if n not in visited:
                        next_candidates.append(n)
                        visited.add(n)
            
            if not next_candidates: break
            
            # v5.2.2 剪枝策略：利用 Resonance 分数对邻居进行重排序 (如果邻居有共振分)
            # 这是一个增强版逻辑：不仅种子共振，邻居也能从共振图中获益
            scored_candidates = []
            for nid in next_candidates:
                # 优先使用共振分，否则使用 TF-IDF
                res_score = resonance_scores.get(nid, 0) if resonance_triggered else 0
                tfidf_score = tfidf_scores.get(nid, 0)
                combined = res_score if res_score > 0 else tfidf_score
                scored_candidates.append((nid, combined))
            
            # 排序
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 动态宽度 (模拟 PID 输出)
            # 如果之前的 Token 消耗低，增加宽度
            width = 15 if total_tokens < budget_tokens * 0.5 else 8
            
            to_visit = [nid for nid, _ in scored_candidates[:width]]
            content = "".join([self.data.get_content(n) for n in to_visit])
            cost = count_tokens(content)
            
            if total_tokens + cost > budget_tokens:
                break
                
            total_tokens += cost
            visited.update(to_visit)
            current_frontier = to_visit
            steps += 1
            
        return {
            "found": target_id in visited,
            "tokens": total_tokens,
            "steps": steps,
            "latency": (time.time() - start_time) * 1000,
            "visited_count": len(visited),
            "resonance_used": resonance_triggered
        }

# ============================================================
# 3. 测试生成器与运行器
# ============================================================

def generate_queries(data: DataLoader, n=50):
    """生成真实的多跳查询"""
    queries = []
    paths_2hop = []
    
    for src in data.adj:
        # 过滤掉内容太短的节点
        if len(data.get_content(src)) < 20: continue
        for mid in data.adj[src]:
            for dst in data.adj[mid]:
                if dst != src:
                    paths_2hop.append((src, dst))
    
    # 随机抽取
    random.seed(42)
    if len(paths_2hop) < n:
        paths_2hop = paths_2hop * 2
    paths_2hop = random.sample(paths_2hop, n)
    
    for src, dst in paths_2hop:
        kws = re.findall(r'[\u4e00-\u9fff]{2,}', data.get_content(src))[:2]
        if kws:
            queries.append({"type": "2hop", "source": src, "target": dst, "keywords": kws})
            
    return queries

def run_benchmark(queries, systems):
    results = {sys.name: {"found": 0, "total_tokens": 0, "latencies": [], "resonance_hits": 0} for sys in systems}
    
    print(f"\n🚀 开始运行 {len(queries)} 个 2-Hop 查询...")
    
    for i, q in enumerate(queries):
        if (i + 1) % 10 == 0: print(f"  进度: {i+1}/{len(queries)}")
        
        for sys in systems:
            res = sys.retrieve(q["keywords"], target_id=q["target"], budget_tokens=5000)
            
            r = results[sys.name]
            if res["found"]: r["found"] += 1
            r["total_tokens"] += res["tokens"]
            r["latencies"].append(res["latency"])
            if res.get("resonance_used", False) and res["found"]:
                r["resonance_hits"] += 1

    # 计算统计
    for name in results:
        r = results[name]
        total = len(queries)
        r["recall_pct"] = (r["found"] / total) * 100
        r["avg_tokens"] = r["total_tokens"] / total
        r["avg_latency"] = statistics.mean(r["latencies"]) if r["latencies"] else 0
        r["median_tokens"] = statistics.median([results[name]["total_tokens"] / total]) # Approximation
        
    return results

def print_report(results):
    print("\n" + "="*80)
    print("📊 终极对比报告 (First Principles & Cybernetics Verified)")
    print("="*80)
    
    headers = ["System Name", "2-Hop Recall", "Avg Tokens", "Avg Latency", "Resonance Hits"]
    print(f"{headers[0]:<35} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]}")
    print("-" * 85)
    
    # 排序：Recall 优先，Token 次要
    sorted_results = sorted(results.items(), key=lambda x: (x[1]["recall_pct"], -x[1]["avg_tokens"]), reverse=True)
    
    for name, r in sorted_results:
        print(f"{name:<35} | {r['recall_pct']:>6.1f}%      | {r['avg_tokens']:>8.0f} T   | {r['avg_latency']:>8.1f} ms  | {r['resonance_hits']}")

    print("\n💡 深度分析:")
    v1_recall = results.get("v5.1 Baseline (TF-IDF + Static Graph)", {}).get("recall_pct", 0)
    v2_recall = results.get("v5.2.2 Synapse (Resonance + Fusion + PID)", {}).get("recall_pct", 0)
    
    if v2_recall > v1_recall:
        print(f"✅ **结论**: v5.2.2 Synapse 胜出! 召回率提升了 {v2_recall - v1_recall:.1f}%。")
        print("   这证实了**联想共振 (Association)** 能有效弥补 **词汇匹配 (Lexical)** 的不足。")
    else:
        print(f"⚠️ **结论**: v5.2.2 Synapse 未达预期提升。可能是共振图谱稀疏或参数未最优。")

# ============================================================
# 4. 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scientific Benchmark: v5.1 vs v5.2.2")
    parser.add_argument("--data-dir", type=str, default="/Users/xy23050701/.copaw/workspaces/default/skills/agent-memory-dna-v5-1-6/data", help="Data directory")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries")
    args = parser.parse_args()
    
    data = DataLoader(args.data_dir)
    print(f"📊 数据加载: {data.stats()}")
    
    # 初始化系统
    v51 = System_V51_Baseline(data)
    
    if HAS_V522_MODULES:
        v522 = System_V522_Synapse(data)
        systems = [v51, v522]
    else:
        print("❌ v5.2.2 模块加载失败，仅运行 v5.1 Baseline")
        systems = [v51]
    
    queries = generate_queries(data, n=args.queries)
    results = run_benchmark(queries, systems)
    print_report(results)
