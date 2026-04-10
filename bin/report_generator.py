#!/usr/bin/env python3
"""
📊 Report Generator - Converts JSON benchmark results to HTML report
"""
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

def generate_html(results_path: str, output_path: str):
    with open(results_path) as f:
        data = json.load(f)
    
    niah = data['niah']
    multi = data['multihop']
    meta = data['metadata']

    # 生成 HTML
    # 为了简洁，这里使用内联样式。
    
    # 颜色方案
    def color(val, type):
        if type == 'recall':
            if val >= 0.8: return '#00ff88'
            if val >= 0.5: return '#88ffaa'
            if val >= 0.2: return '#ffd93d'
            return '#ff6b6b'
        elif type == 'token':
            if val < 500: return '#00ff88'
            if val < 1500: return '#88ffaa'
            if val < 5000: return '#ffd93d'
            return '#ff6b6b'
        return '#ccc'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Memory System Benchmark Report</title>
<style>
body {{ font-family: sans-serif; background: #0f0c29; color: #e0e0e0; padding: 40px; }}
.container {{ max-width: 1000px; margin: 0 auto; background: #1a1a3a; padding: 30px; border-radius: 10px; }}
h1 {{ color: #00d2ff; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #2a2a5a; }}
th, td {{ padding: 12px; border-bottom: 1px solid #444; text-align: center; }}
th {{ background: #3a3a7a; }}
.bar {{ height: 10px; background: #555; border-radius: 5px; overflow: hidden; margin-top: 5px; }}
.bar-fill {{ height: 100%; border-radius: 5px; }}
.footer {{ margin-top: 30px; color: #888; font-size: 0.9em; text-align: center; }}
</style>
</head>
<body>
<div class="container">
    <h1>🔬 记忆系统科学对比测试报告</h1>
    <p>📊 数据集: {meta['dataset']['nodes']} 节点 | 查询: {meta['queries']} 次/类</p>
    
    <h2>🧪 NIAH (召回率)</h2>
    <table>
    <tr><th>系统</th><th>召回率</th><th>Token (均值)</th><th>延迟 (ms)</th></tr>
    """
    
    for name, d in sorted(niah.items(), key=lambda x: x[1]['recall'], reverse=True):
        c = color(d['recall'], 'recall')
        tc = color(d['avg_tokens'], 'token')
        pct = d['recall'] * 100
        html += f"""
    <tr>
        <td style="text-align:left; padding-left:20px;">{name}</td>
        <td style="color:{c}; font-weight:bold;">{d['recall_pct']}</td>
        <td style="color:{tc};">{d['avg_tokens']:.0f}</td>
        <td>{d['avg_latency']:.2f}</td>
    </tr>
    """
    
    html += f"""
    </table>

    <h2>🔗 MultiHop (多跳推理)</h2>
    <table>
    <tr><th>系统</th><th>2-Hop</th><th>3-Hop</th><th>Token (均值)</th></tr>
    """

    for name, d in sorted(multi.items(), key=lambda x: x[1]['2hop_recall'], reverse=True):
        c2 = color(d['2hop_recall'], 'recall')
        c3 = color(d['3hop_recall'], 'recall')
        tc = color(d['avg_tokens'], 'token')
        html += f"""
    <tr>
        <td style="text-align:left; padding-left:20px;">{name}</td>
        <td style="color:{c2}; font-weight:bold;">{d['2hop_recall_pct']}</td>
        <td style="color:{c3}; font-weight:bold;">{d['3hop_recall_pct']}</td>
        <td style="color:{tc};">{d['avg_tokens']:.0f}</td>
    </tr>
    """

    html += f"""
    </table>
    
    <div class="footer">Generated on {datetime.now().strftime("%Y-%m-%d")} | Token Counter: {meta['token_counter']}</div>
</div>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"✅ HTML 报告已生成: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to benchmark_results.json")
    parser.add_argument("--output", type=str, default="benchmark_report.html", help="Output HTML file")
    args = parser.parse_args()
    generate_html(args.results, args.output)
