#!/usr/bin/env python
"""Write a synthetic needle-in-a-haystack prompt set as JSONL for capture.py.

The haystack is generated, deterministic filler (no download). Each prompt hides
one fact at a given depth and asks for it at the end. Lengths are approximate
token counts for the chosen tokenizer.
"""

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

SUBJECTS = ["The harbour town", "A mountain village", "The old library", "The river ferry", "A desert outpost",
            "The northern mill", "The glass foundry", "A coastal lighthouse", "The orchard estate", "The rail depot"]
VERBS = ["kept records of", "was known for", "traded mostly in", "was rebuilt after", "held a yearly fair for",
         "taxed heavily", "was famous for", "slowly lost", "grew rich on", "argued for decades about"]
OBJECTS = ["salted fish", "copper wire", "wool blankets", "a flood in spring", "its clock tower", "migrating geese",
           "river tolls", "hand-bound maps", "a failed harvest", "imported tea"]


def haystack(rng, n_sentences):
    return " ".join(
        f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)} in the year {rng.randint(1500, 1950)}."
        for _ in range(n_sentences)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096])
    p.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    rng = random.Random(args.seed)
    tokens_per_sentence = len(tok(haystack(random.Random(1), 200))["input_ids"]) / 200
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for length in args.lengths:
            for depth in args.depths:
                code = rng.randint(10000, 99999)
                sentences = haystack(rng, int(length / tokens_per_sentence)).split(". ")
                at = int(depth * len(sentences))
                sentences.insert(at, f"The secret access code for the vault is {code}")
                text = ". ".join(sentences)
                prompt = f"{text}\n\nWhat is the secret access code for the vault? Answer with the number only."
                rec = {"id": f"needle_{length}_d{int(depth * 100):02d}", "prompt": prompt, "answer": str(code)}
                f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(args.lengths) * len(args.depths)} prompts to {args.out}")


if __name__ == "__main__":
    main()
