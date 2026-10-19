import pandas as pd
from simulator import run_experiment

PROMPT_LENGTH = 4000
CHUNK_SIZES = [250, 500, 1000, 2000, 4000]

results = []

for chunk_size in CHUNK_SIZES:
    # A chunk size equal to the full prompt represents normal prefill.
    mode = "normal" if chunk_size == PROMPT_LENGTH else "chunked"

    summary, events, decode_metrics = run_experiment(
        mode=mode,
        prompt_length=PROMPT_LENGTH,
        chunk_size=chunk_size,
    )

    summary["tested_chunk_size"] = chunk_size
    results.append(summary)

    print(
        f"Tested chunk size {chunk_size}: "
        f"TTFT={summary['new_prompt_ttft']:.3f}s, "
        f"decode first-token latency="
        f"{summary['average_decode_first_token_latency']:.3f}s, "
        f"throughput="
        f"{summary['aggregate_token_throughput']:.2f} tokens/s"
    )

df = pd.DataFrame(results)

df.to_csv("results/chunk_size_experiments.csv", index=False)

print("\nCHUNK SIZE EXPERIMENT RESULTS")
print("=" * 100)
print(df[
    [
        "mode",
        "tested_chunk_size",
        "new_prompt_ttft",
        "average_decode_first_token_latency",
        "average_inter_token_latency",
        "aggregate_token_throughput",
    ]
].to_string(index=False))

print("\nResults saved to results/chunk_size_experiments.csv")
