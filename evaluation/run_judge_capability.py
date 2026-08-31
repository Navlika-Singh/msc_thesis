"""
Judge-only pass for the accuracy judge: given a JSONL of already-generated
policy responses (as produced by run_eval.py -- one record per line with
id/benchmark/category/question/thinking/answer), re-run the accuracy judge
(judge_accuracy.judge_batch) without touching the policy model at all.

This is the accuracy-metric counterpart to run_judge.py -- same loading /
image re-attachment / resume logic, but grades each response against a
ground-truth answer (Is_Correct) instead of scoring safety/helpfulness/
reasoning-rigor, and reports overall accuracy instead of win rates.

Since generation output doesn't carry the original PIL image with it,
--data must point at the same source used for run_eval.py's --data so
images can be re-attached by sample id (same dataset id + split + dedup
settings => same ids, deterministically). --data must also be the source
that carries the ground-truth answer field(s) (sample.answer / gt_answer /
ground_truth) and, for MCQ samples, sample.choices.

Usage:
    python run_judge_accuracy.py \
        --generations outputs/results.jsonl \
        --data your/dataset-with-ground-truth --split evaluation \
        --judge qwen3-vl-32b-instruct \
        --batch-size 8 \
        --out outputs/results_judged_accuracy.jsonl
"""
import argparse
import json

from dataset import Sample, load_for_data_arg
from judge_capability import compute_accuracy, judge_batch
from models import DEFAULT_JUDGE, LoadedModel
from resume import load_done_ids, load_existing_records, open_output


def load_generations(path: str) -> list:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generations",
        required=True,
        help="Path to a JSONL of pre-generated responses (id, question, thinking, answer, ...)",
    )
    ap.add_argument(
        "--data",
        required=True,
        help="Same --data source used to produce --generations (JSONL path or HF dataset id), "
        "used to re-attach images AND ground-truth answers/choices by sample id",
    )
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="Judge model key, see models.JUDGE_REGISTRY")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--judge-max-new-tokens", type=int, default=512)
    ap.add_argument("--out", default="outputs/judged_accuracy.jsonl")
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable resume: truncate --out and re-judge everything",
    )
    args = ap.parse_args()

    gens = load_generations(args.generations)
    print(f"[INFO] Loaded {len(gens)} generations from {args.generations}")

    ref_samples = load_for_data_arg(args.data, args.split)
    by_id = {s.id: s for s in ref_samples}

    missing = [g["id"] for g in gens if g["id"] not in by_id]
    if missing:
        print(
            f"[WARN] {len(missing)} generation ids not found in --data "
            f"(dataset/split/dedup mismatch with the original run?); skipping them. "
            f"First few: {missing[:5]}"
        )
        gens = [g for g in gens if g["id"] in by_id]

    samples, thinking, answers = [], [], []
    for g in gens:
        ref = by_id[g["id"]]
        samples.append(
            Sample(
                id=ref.id,
                benchmark=g.get("benchmark", ref.benchmark),
                question=g.get("question", ref.question),
                image=ref.image,
                image_path=ref.image_path,
                category=g.get("category", ref.category),
                answer=g.get("answer", ref.answer)
            )
        )
        thinking.append(g.get("thinking", ""))
        answers.append(g.get("answer", ""))

    done_ids = load_done_ids(args.out) if args.resume else set()
    if done_ids:
        print(f"[INFO] Resuming from {args.out}: {len(done_ids)} already judged, skipping them")
    records = load_existing_records(args.out) if done_ids else []

    keep = [i for i, s in enumerate(samples) if s.id not in done_ids]
    samples = [samples[i] for i in keep]
    thinking = [thinking[i] for i in keep]
    answers = [answers[i] for i in keep]

    if not samples:
        print("[INFO] Nothing left to judge -- all generations already present in --out")
    else:
        judge = LoadedModel.judge(args.judge)
        with open_output(args.out, append=bool(done_ids)) as out_f:
            for i in range(0, len(samples), args.batch_size):
                batch_s = samples[i : i + args.batch_size]
                batch_t = thinking[i : i + args.batch_size]
                batch_a = answers[i : i + args.batch_size]

                scores = judge_batch(
                    judge, batch_s, batch_t, batch_a, max_new_tokens=args.judge_max_new_tokens
                )

                for s, t, a, sc in zip(batch_s, batch_t, batch_a, scores):
                    record = {
                        "id": s.id,
                        "benchmark": s.benchmark,
                        "category": s.category,
                        "question": s.question,
                        "thinking": t,
                        "answer": a,
                        **sc,
                    }
                    records.append(record)
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                print(f"[INFO] {len(records)} judged")

    agg = compute_accuracy(records)
    print(
        f"[RESULT] accuracy={agg['accuracy']:.4f} "
        f"({agg['correct']}/{agg['judged']} correct, {agg['unparsed']} unparsed, {agg['total']} total)"
    )
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()