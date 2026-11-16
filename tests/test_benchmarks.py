"""Tests for the scorers and RULER parsing in phase0/benchmarks.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
import benchmarks as b  # noqa: E402


def test_qa_f1():
    assert b.score("qa_f1", "The cat sat.", ["cat sat"]) == 1.0
    assert b.score("qa_f1", "a dog", ["cat sat", "dog"]) == 1.0
    assert b.score("qa_f1", "cat", ["cat sat"]) == pytest.approx(2 / 3)


def test_first_line_rule_for_fewshot_tasks():
    assert b.score("qa_f1", "\nParis\nLondon is also big", ["Paris"], task="triviaqa") == 1.0
    assert b.score("qa_f1", "\nParis\nLondon is also big", ["Paris"]) < 1.0


def test_classification_reproduces_longbench():
    classes = ["Entity", "Location", "Location city"]
    assert b.score("classification", "Location city", ["Location city"], classes) == 1.0
    assert b.score("classification", "Entity or Location", ["Location"], classes) == 0.5


def test_retrieval_and_count():
    assert b.score("retrieval", "Paragraph 12", ["Paragraph 12"]) == 1.0
    assert b.score("retrieval", "Paragraph 3 or 12", ["Paragraph 12"]) == 0.5
    assert b.score("count", "There are 7 unique paragraphs", ["7"]) == 1.0


def test_code_sim_skips_comment_lines():
    assert b.score("code_sim", "# comment\nreturn x + 1", ["return x + 1"]) == 1.0


def test_rouge_identical_and_disjoint():
    assert b.score("rouge", "the report finds growth. costs fell.", ["the report finds growth. costs fell."]) == pytest.approx(1.0)
    assert b.score("rouge", "alpha beta", ["gamma delta"]) == 0.0


def test_ruler_string_match():
    assert b.score("string_match_all", " 123, 456\n", ["123", "456"]) == 1.0
    assert b.score("string_match_all", "123", ["123", "456"]) == 0.5
    assert b.score("string_match_part", "it is 456", ["123", "456"]) == 1.0


def test_split_llama3():
    text = ("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n Find the code. Text...<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n Answer: The code is")
    content, prefix = b.split_llama3(text)
    assert content == "Find the code. Text..." and prefix == "Answer: The code is"
