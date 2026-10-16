"""Command line interface to the NLP component; every command prints JSON.

  python -m nlp.component analyze "I lost my ID card, roll no J054"
  python -m nlp.component intent|entities|retrieve "..."
  python -m nlp.component batch queries.txt          # one query per line ('-' reads stdin); prints JSON lines
  python -m nlp.component info
  python -m nlp.component schema [out_dir]           # JSON Schemas of all inputs/outputs (default docs/schemas/)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nlp import PROJECT_ROOT


def write_schemas(out: Path) -> list[Path]:
    from nlp.component.schemas import ALL_SCHEMAS

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in ALL_SCHEMAS:
        p = out / f"{model.__name__}.json"
        p.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nlp.component", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyze", "intent", "entities", "retrieve"):
        s = sub.add_parser(name)
        s.add_argument("text")
        s.add_argument("--k", type=int, default=3, help="passages (analyze, retrieve) / alternatives (intent)")
    b = sub.add_parser("batch")
    b.add_argument("file", help="text file, one query per line; '-' for stdin")
    sub.add_parser("info")
    sc = sub.add_parser("schema")
    sc.add_argument("out", nargs="?", default=str(PROJECT_ROOT / "docs" / "schemas"))
    a = ap.parse_args(argv)

    if a.cmd == "schema":
        for p in write_schemas(Path(a.out)):
            print(p)
        return 0

    from pydantic import ValidationError

    from nlp.component import AdminNLP, ModelsUnavailable
    from nlp.component.schemas import AnalyzeInput, IntentInput, Options, RetrieveInput

    nlp = AdminNLP()
    try:
        if a.cmd == "batch":
            lines = (sys.stdin if a.file == "-" else open(a.file, encoding="utf-8")).read().splitlines()
            for n, line in enumerate(l for l in lines if l.strip()):
                print(nlp.analyze(AnalyzeInput(text=line, request_id=str(n + 1))).model_dump_json())
            return 0
        out = {
            "analyze": lambda: nlp.analyze(AnalyzeInput(text=a.text, options=Options(k_sources=a.k))),
            "intent": lambda: nlp.classify(IntentInput(text=a.text, top_k=a.k)),
            "entities": lambda: nlp.extract_entities(a.text),
            "retrieve": lambda: nlp.retrieve(RetrieveInput(text=a.text, k=a.k)),
            "info": nlp.info,
        }[a.cmd]()
    except ValidationError as e:
        print(json.dumps({"error": "validation_error", "detail": str(e)}), file=sys.stderr)
        return 2
    except ModelsUnavailable as e:
        print(json.dumps({"error": "models_unavailable", "detail": str(e)}), file=sys.stderr)
        return 3
    print(out.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
