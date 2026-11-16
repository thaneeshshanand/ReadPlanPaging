#!/usr/bin/env python
"""Build Phase 0 prompt sets from LongBench and RULER, and score predictions.

Writes JSONL records that capture.py and tiered.py read directly:

    id, prompt, answers, task, metric, max_new_tokens, source, n_tokens,
    chat_template (false for LongBench few-shot and code tasks, as in LongBench's pred.py),
    assistant_prefix (RULER answer prefixes), all_classes (LongBench classification)

n_tokens is the prompt length with the target tokenizer, chat template and assistant
prefix included, i.e. exactly what capture.py feeds the model. Records are kept when
min_tokens <= n_tokens <= max_tokens, sampled per task with a fixed seed.

Sources (pinned):
- LongBench v1: data.zip from the Hugging Face dataset THUDM/LongBench; prompt templates
  and output lengths from github.com/THUDM/LongBench (LongBench/config) at a fixed commit.
- RULER: pre-generated tasks from the dataset self-long/RULER-llama3-1M, made with the
  official RULER scripts and the Llama-3 chat template. The Llama-3 template is stripped
  and the content re-templated for the target model, keeping RULER's answer prefix.
  RULER's own lengths count Llama-3 tokens; n_tokens here is re-measured.

Scoring follows LongBench's metrics.py/eval.py and RULER's eval constants, including
LongBench's first-line rule for trec/triviaqa/samsum and its classification quirk.
Rouge-L uses the `rouge` package and code similarity uses `fuzzywuzzy` when installed;
otherwise pure-Python stand-ins are used (summary-level Rouge-L, difflib ratio), and
score_backends() says which.

Usage:
    python phase0/benchmarks.py --tokenizer meta-llama/Llama-3.1-8B-Instruct \\
        --out data/prompts/llama8b_8k-32k.jsonl --min-tokens 8000 --max-tokens 32000 \\
        --ruler-lengths 8k 16k 32k --per-task 3
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import re
import string
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

LONGBENCH_REPO = "THUDM/LongBench"
LONGBENCH_HF_REVISION = "5e628be450"  # resolved to the full sha at download time
LONGBENCH_GH_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
RULER_REPO = "self-long/RULER-llama3-1M"
RULER_GH_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"  # metric/length constants below

LONGBENCH_METRIC = {
    "narrativeqa": "qa_f1", "qasper": "qa_f1", "multifieldqa_en": "qa_f1", "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1", "musique": "qa_f1", "gov_report": "rouge", "qmsum": "rouge", "multi_news": "rouge",
    "trec": "classification", "triviaqa": "qa_f1", "samsum": "rouge", "passage_retrieval_en": "retrieval",
    "passage_count": "count", "lcc": "code_sim", "repobench-p": "code_sim",
}
LONGBENCH_NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}
DEFAULT_LONGBENCH = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
                     "gov_report", "qmsum", "trec", "triviaqa", "samsum", "passage_retrieval_en",
                     "passage_count", "lcc", "repobench-p"]

RULER_GEN = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
RULER_METRIC = {"niah": "string_match_all", "vt": "string_match_all", "cwe": "string_match_all",
                "fwe": "string_match_all", "qa": "string_match_part"}
DEFAULT_RULER = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
                 "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"]


# --------------------------------------------------------------------------- scoring


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred_tokens, gt_tokens) -> float:
    common = Counter(pred_tokens) & Counter(gt_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_tokens), same / len(gt_tokens)
    return 2 * p * r / (p + r)


def qa_f1(pred, gt, **_):
    return _f1(normalize_answer(pred).split(), normalize_answer(gt).split())


def count_score(pred, gt, **_):
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == str(gt) for n in nums) / len(nums)


def retrieval_score(pred, gt, **_):
    gid = re.findall(r"Paragraph (\d+)", gt)[0]
    nums = re.findall(r"\d+", pred)
    return 0.0 if not nums else sum(n == gid for n in nums) / len(nums)


def classification_score(pred, gt, all_classes=None, **_):
    matches = [c for c in all_classes if c in pred]
    for term in matches:  # LongBench removes from the list while iterating; reproduced on purpose
        if term in gt and term != gt:
            matches.remove(term)
    return 1.0 / len(matches) if gt in matches else 0.0


def _rouge_l_fallback(pred: str, gt: str) -> float:
    """Summary-level Rouge-L F1 in the style of the `rouge` package (sentences split on '.')."""
    hyp = [s.split() for s in pred.split(".") if s.strip()]
    ref = [s.split() for s in gt.split(".") if s.strip()]
    if not hyp or not ref:
        return 0.0
    m = sum(len(r) for r in ref)
    n = sum(len(h) for h in hyp)
    hits = 0
    for r in ref:  # union LCS per reference sentence
        hit = set()
        for h in hyp:
            # positions of r covered by an LCS with h
            dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
            for i in range(len(r)):
                for j in range(len(h)):
                    dp[i + 1][j + 1] = dp[i][j] + 1 if r[i] == h[j] else max(dp[i][j + 1], dp[i + 1][j])
            i, j = len(r), len(h)
            while i and j:
                if r[i - 1] == h[j - 1]:
                    hit.add(i - 1)
                    i, j = i - 1, j - 1
                elif dp[i - 1][j] >= dp[i][j - 1]:
                    i -= 1
                else:
                    j -= 1
        hits += len(hit)
    if hits == 0:
        return 0.0
    p, rcl = hits / n, hits / m
    return 2 * p * rcl / (p + rcl)


try:
    from rouge import Rouge as _Rouge  # type: ignore

    def rouge_score(pred, gt, **_):
        try:
            return _Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]
        except Exception:
            return 0.0

    _ROUGE = "rouge package"
except ImportError:
    def rouge_score(pred, gt, **_):
        return _rouge_l_fallback(pred, gt)

    _ROUGE = "fallback (pure-Python summary-level Rouge-L)"

try:
    from fuzzywuzzy import fuzz as _fuzz  # type: ignore

    _ratio = lambda a, b: _fuzz.ratio(a, b)
    _FUZZ = "fuzzywuzzy"
except ImportError:
    _ratio = lambda a, b: round(100 * difflib.SequenceMatcher(None, a, b).ratio())
    _FUZZ = "fallback (difflib, as fuzzywuzzy without python-Levenshtein)"


def code_sim(pred, gt, **_):
    line = next((l for l in pred.lstrip("\n").split("\n") if "`" not in l and "#" not in l and "//" not in l), "")
    return _ratio(line, gt) / 100


def string_match_all(pred, answers):
    pred = _ruler_post(pred).lower()
    return sum(a.lower() in pred for a in answers) / len(answers)


def string_match_part(pred, answers):
    pred = _ruler_post(pred).lower()
    return max(float(a.lower() in pred) for a in answers)


def _ruler_post(pred: str) -> str:
    return re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip()


PER_ANSWER = {"qa_f1": qa_f1, "rouge": rouge_score, "classification": classification_score,
              "retrieval": retrieval_score, "count": count_score, "code_sim": code_sim}


def score(metric: str, pred: str, answers: list[str], all_classes=None, task: str | None = None) -> float:
    """Score in [0, 1] for one prediction, as LongBench / RULER do (they report x100)."""
    if metric == "string_match_all":
        return string_match_all(pred, answers)
    if metric == "string_match_part":
        return string_match_part(pred, answers)
    if task in FIRST_LINE_TASKS:
        pred = pred.lstrip("\n").split("\n")[0]
    fn = PER_ANSWER[metric]
    return max(fn(pred, gt, all_classes=all_classes) for gt in answers)


def score_backends() -> dict:
    return {"rouge": _ROUGE, "code_sim": _FUZZ}


# --------------------------------------------------------------------------- sources


def _fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as r:
            dest.write_bytes(r.read())
    return dest


def longbench_files() -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    sha = HfApi().dataset_info(LONGBENCH_REPO, revision=LONGBENCH_HF_REVISION).sha
    zpath = Path(hf_hub_download(LONGBENCH_REPO, "data.zip", repo_type="dataset", revision=sha))
    out = RAW / "longbench" / sha
    if not (out / "data").exists():
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():  # extract only plain relative paths
                if m.startswith("/") or ".." in Path(m).parts:
                    raise ValueError(f"unsafe path in zip: {m}")
            z.extractall(out)
    gh = f"https://raw.githubusercontent.com/THUDM/LongBench/{LONGBENCH_GH_COMMIT}/LongBench/config"
    prompts = _fetch(f"{gh}/dataset2prompt.json", RAW / "longbench" / LONGBENCH_GH_COMMIT / "dataset2prompt.json")
    maxlen = _fetch(f"{gh}/dataset2maxlen.json", RAW / "longbench" / LONGBENCH_GH_COMMIT / "dataset2maxlen.json")
    return {"data": out / "data", "prompts": prompts, "maxlen": maxlen, "hf_sha": sha}


def longbench_records(task: str, files: dict):
    templates = json.loads(files["prompts"].read_text())
    maxlen = json.loads(files["maxlen"].read_text())
    with open(files["data"] / f"{task}.jsonl") as f:
        for line in f:
            d = json.loads(line)
            yield {
                "id": f"lb-{task}-{d['_id'][:12]}",
                "prompt": templates[task].format(**d),
                "chat_template": task not in LONGBENCH_NO_CHAT,
                "answers": d["answers"],
                "all_classes": d.get("all_classes"),
                "task": task,
                "metric": LONGBENCH_METRIC[task],
                "max_new_tokens": maxlen[task],
                "source": f"LongBench@{files['hf_sha'][:10]}",
                "source_length": d.get("length"),
            }


_L3_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
_L3_ASSIST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def split_llama3(text: str) -> tuple[str, str]:
    """User content and assistant prefix from a Llama-3-templated RULER input."""
    if _L3_USER not in text or _L3_ASSIST not in text:
        raise ValueError("input does not look Llama-3-templated")
    user = text.split(_L3_USER, 1)[1]
    content, prefix = user.rsplit(_L3_ASSIST, 1)
    return content.strip(), prefix.strip()


def ruler_family(task: str) -> str:
    return task.split("_")[0] if task.startswith(("niah", "qa")) else task


def ruler_records(task: str, length: str):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    sha = HfApi().dataset_info(RULER_REPO).sha
    path = hf_hub_download(RULER_REPO, f"{task}_{length}/validation-00000-of-00001.parquet", repo_type="dataset", revision=sha)
    fam = ruler_family(task)
    for d in pq.read_table(path, columns=["index", "input", "answers", "length"]).to_pylist():
        content, prefix = split_llama3(d["input"])
        yield {
            "id": f"ruler-{task}-{length}-{d['index']}",
            "prompt": content,
            "assistant_prefix": prefix,
            "chat_template": True,
            "answers": list(d["answers"]),
            "task": f"ruler_{task}",
            "metric": RULER_METRIC[fam],
            "max_new_tokens": RULER_GEN[fam],
            "source": f"{RULER_REPO}@{sha[:10]}",
            "source_length": d["length"],
        }


# --------------------------------------------------------------------------- build


def prompt_tokens(tok, rec) -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capture import encode

    ids, _ = encode(tok, rec, use_template=True, max_context=10**9)
    return len(ids)


def sample(records, tok, n, lo, hi, rng):
    recs = list(records)
    rng.shuffle(recs)
    kept = []
    for r in recs:
        r["n_tokens"] = prompt_tokens(tok, r)
        if lo <= r["n_tokens"] <= hi:
            kept.append(r)
            if len(kept) == n:
                break
    return kept


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True, help="target model's tokenizer (lengths and templates)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-tokens", type=int, default=8000)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--per-task", type=int, default=3)
    p.add_argument("--longbench-tasks", nargs="*", default=DEFAULT_LONGBENCH)
    p.add_argument("--ruler-tasks", nargs="*", default=DEFAULT_RULER)
    p.add_argument("--ruler-lengths", nargs="*", default=["8k", "16k", "32k"],
                   help="RULER configs to draw from (Llama-3 token lengths: 4k 8k 16k 32k ...)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    out, report = [], []
    if args.longbench_tasks:
        files = longbench_files()
        for task in args.longbench_tasks:
            kept = sample(longbench_records(task, files), tok, args.per_task, args.min_tokens, args.max_tokens, rng)
            out += kept
            report.append((f"longbench/{task}", len(kept)))
    for task in args.ruler_tasks:
        recs = [r for L in args.ruler_lengths for r in ruler_records(task, L)]
        kept = sample(recs, tok, args.per_task, args.min_tokens, args.max_tokens, rng)
        out += kept
        report.append((f"ruler/{task}", len(kept)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "args": {k: str(v) for k, v in vars(args).items()},
        "records": len(out), "sha256": digest, "per_task": dict(report),
        "sources": {"longbench_hf": LONGBENCH_REPO, "longbench_github_commit": LONGBENCH_GH_COMMIT,
                    "ruler_hf": RULER_REPO, "ruler_github_commit_for_constants": RULER_GH_COMMIT},
        "score_backends": score_backends(),
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    for name, k in report:
        flag = "" if k == args.per_task else f"  (only {k} in range)"
        print(f"{name:32} {k}{flag}")
    print(f"{len(out)} records -> {args.out}")


if __name__ == "__main__":
    main()
