"""Retrieval/answer quality check against a small question set.

Usage (from the project root, with Ollama running):
    python -m evaluation.evaluate evaluation/sample_questions.json

Each case: {"question": "...", "expect_keywords": ["..."], "expect_document": "file.pdf",
            "expect_insufficient": false}
Retrieval is scored without the LLM; answers are checked only if Ollama is up.
"""
import json
import sys

from ai_engine.prompt_builder import INSUFFICIENT_INFO
from pipeline import KnowledgeAssistant


def run(cases_path):
    assistant = KnowledgeAssistant(extract_concepts=False)
    assistant.index_embeddings()
    cases = json.loads(open(cases_path, encoding="utf-8").read())
    llm_up = assistant.llm.is_available()
    hits = answered = 0

    for case in cases:
        chunks = assistant.retrieve(case["question"])
        text = " ".join(c["text"].lower() for c in chunks)
        if case.get("expect_insufficient"):
            ok = not chunks
        else:
            ok = all(k.lower() in text for k in case.get("expect_keywords", []))
            if case.get("expect_document"):
                ok = ok and any(c["document_name"] == case["expect_document"] for c in chunks)
        hits += ok
        line = f"{'PASS' if ok else 'FAIL'} retrieval | {case['question']}"

        if llm_up:
            result = assistant.ask(case["question"])
            if case.get("expect_insufficient"):
                good = result["answer"].strip() == INSUFFICIENT_INFO
            else:
                good = not result["error"] and any(s["cited"] for s in result["sources"])
            answered += good
            line += f" | answer {'PASS' if good else 'FAIL'}"
        print(line)

    print(f"\nRetrieval: {hits}/{len(cases)}")
    if llm_up:
        print(f"Answers:   {answered}/{len(cases)}")
    else:
        print("Answers:   skipped (Ollama not running)")
    return hits == len(cases)


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1] if len(sys.argv) > 1 else "evaluation/sample_questions.json") else 1)
