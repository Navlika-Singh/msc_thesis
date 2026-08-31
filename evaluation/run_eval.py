"""
python run_eval.py \
    --data PKU-Alignment/BeaverTails-V --split evaluation \
    --policy qwen2.5-vl-7b-instruct \
    --judge qwen3-vl-32b-instruct \
    --batch-size 8 \
    --out outputs/results.jsonl
"""
import argparse
import json

from aggregate import aggregate, print_win_rate_table
from dataset import batched, load_for_data_arg
from generate import generate_batch, parse_thinking_answer
# from judge import judge_batch
from models import LoadedModel
from resume import load_done_ids, load_existing_records, open_output
from prompt_strategies import STRATEGIES, get_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="Path to a JSONL eval set, OR a HF dataset id (e.g. PKU-Alignment/BeaverTails-V) to load via datasets",
    )
    ap.add_argument(
        "--split",
        default="evaluation",
        help="Split to use when --data is a HF dataset id (default: evaluation)",
    )
    ap.add_argument("--policy", required=True, help="Policy model key, see models.POLICY_REGISTRY")
    # ap.add_argument("--judge", default=DEFAULT_JUDGE, help="Judge model key, see models.JUDGE_REGISTRY")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    # ap.add_argument("--judge-max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default="outputs/results.jsonl")
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable resume: truncate --out and start over instead of skipping already-done ids",
    )
    ap.add_argument("--strategy", default="direct", choices=list(STRATEGIES))
    args = ap.parse_args()

    samples = load_for_data_arg(args.data, args.split)
    print(f"[INFO] Loaded {len(samples)} samples from {args.data} (split={args.split})")

    done_ids = load_done_ids(args.out) if args.resume else set()
    if done_ids:
        print(f"[INFO] Resuming from {args.out}: {len(done_ids)} samples already done, skipping them")
    records = load_existing_records(args.out) if done_ids else []
    samples = [s for s in samples if s.id not in done_ids]

    if not samples:
        print("[INFO] Nothing left to do -- all samples already present in --out")
    else:
        policy = LoadedModel.policy(args.policy)
        # judge = LoadedModel.judge(args.judge)

        with open_output(args.out, append=bool(done_ids)) as out_f:
            for batch in batched(samples, args.batch_size):
                print("Generating batch...")
                gens = generate_batch(policy, batch, args.strategy, max_new_tokens=args.max_new_tokens)
                print("Parsing batch...")
                # single greedy rollout per sample -> take [0]
                parsed = [parse_thinking_answer(g[0]) for g in gens]
                thinking = [p["thinking"] for p in parsed]
                answers = [p["answer"] for p in parsed]

                # print("Judge scoring...")
                # scores = judge_batch(
                #     judge, batch, thinking, answers, max_new_tokens=args.judge_max_new_tokens
                # )
                # print("Judge scored...")

                for s, p in zip(batch, parsed):
                    record = {
                        "id": s.id,
                        "benchmark": s.benchmark,
                        "category": s.category,
                        "question": s.question,
                        "thinking": p["thinking"],
                        "answer": p["answer"],
                    }
                    records.append(record)
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                print(f"[INFO] {len(records)} samples done")

    # agg = aggregate(records)
    # print_table(agg)
    # with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
    #     json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()
