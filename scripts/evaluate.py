"""Evaluate real retrieval on a fixed set; --with-llm also calls the configured model."""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import ROOT, settings
from app.conversation.engine import SupportEngine
from app.models import Message
from app.knowledge.retrieval import RetrievalRegistry


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    config = settings()
    registry = RetrievalRegistry(config)
    engine = SupportEngine(config, registry)
    questions = json.loads((ROOT / "evaluation/questions.json").read_text(encoding="utf-8"))
    report = {"dataset": "20 self-authored synthetic product-support questions",
              "scope": "小型模拟资料集结果，不代表通用准确率；关键词判定需人工复核。",
              "retrieval": {}, "answers": [], "model": config.llm_model, "thinking": config.llm_thinking}
    for backend in ("bm25", "tfidf"):
        rows = []
        for index, case in enumerate(questions):
            history = [Message(id=f"h{i}", session_id="eval", turn_id="h", role="user", text=t,
                               created_at="2026-10-14T00:00:00Z") for i, t in enumerate(case.get("history", []))]
            prepared = await engine.prepare(case["question"], history, backend)
            hit = any(c.heading == case.get("heading") for c in prepared.result.chunks) if "heading" in case else None
            rows.append({"question": case["question"], "expected_heading": case.get("heading"),
                         "hit_at_3": hit, "query": prepared.query,
                         "retrieved_headings": [c.heading for c in prepared.result.chunks],
                         "latency_ms": prepared.result.latency_ms})
            if args.with_llm and backend == "bm25":
                try:
                    text, sources, mode, metrics = await engine.answer(prepared, history, case["question"])
                    if case.get("expect") == "unknown":
                        passed = not sources and any(w in text for w in ("无法", "不能确认", "未提及", "没有", "未找到"))
                    elif case.get("expect") == "clarify":
                        passed = metrics.get("answer_type") == "clarify"
                    else:
                        passed = bool(sources) and any(w.lower() in text.lower() for w in case.get("keywords", []))
                    row = {"question": case["question"], "answer": text, "passed_heuristic": passed,
                           "sources": [c.document_name for c in sources], "mode": mode, "metrics": metrics}
                except Exception as exc:
                    row = {"question": case["question"], "passed_heuristic": False, "error": type(exc).__name__}
                report["answers"].append(row)
                print(f"Answer {index + 1}/{len(questions)}: {'PASS' if row['passed_heuristic'] else 'CHECK'}", flush=True)
        known = [r for r in rows if r["hit_at_3"] is not None]
        latencies = sorted(r["latency_ms"] for r in rows)
        report["retrieval"][backend] = {"known_questions": len(known), "hit_at_3": sum(r["hit_at_3"] for r in known),
                                         "p50_ms": statistics.median(latencies), "p95_ms": latencies[int(len(latencies) * .95) - 1], "cases": rows}
    path = ROOT / "evaluation/results.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 固定问题集评估", "", "数据：20 个自建模拟产品问题。结果仅描述此资料集。", "",
             "| 检索实现 | 已知问题命中@3 | 检索 P50 | 检索 P95 |", "|---|---:|---:|---:|"]
    for backend, result in report["retrieval"].items():
        lines.append(f"| {backend} | {result['hit_at_3']}/{result['known_questions']} | {result['p50_ms']:.2f} ms | {result['p95_ms']:.2f} ms |")
    if args.with_llm:
        passed = sum(r["passed_heuristic"] for r in report["answers"])
        lines.extend(["", f"模型：{config.llm_model}；思考：{config.llm_thinking}。", "",
                      f"答案/引用/拒答启发式检查：{passed}/{len(questions)}。关键词检查不能代替人工审阅，完整输出见 results.json。", ""])
        for index, row in enumerate(report["answers"]):
            lines.extend([f"## {index + 1}. {row['question']}", "", row.get("answer", row.get("error", "")), "",
                          "来源：" + "、".join(row.get("sources", [])), ""])
    (ROOT / "evaluation/results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
