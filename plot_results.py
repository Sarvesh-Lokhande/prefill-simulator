import os
import pandas as pd
import matplotlib.pyplot as plt

# Create the graphs directory if it does not exist
os.makedirs("graphs", exist_ok=True)

# Load experiment results
df = pd.read_csv("results/decode_comparison.csv")

# Consistent labels for the plots
labels = ["Normal Prefill", "Chunked Prefill"]

# Metrics to visualize
plots = [
    {
        "column": "new_prompt_ttft",
        "title": "New Prompt Time to First Token (TTFT)",
        "ylabel": "TTFT (seconds)",
        "filename": "ttft_comparison.png",
    },
    {
        "column": "average_decode_first_token_latency",
        "title": "First-Token Latency for Existing Decode Users",
        "ylabel": "Latency (seconds)",
        "filename": "decode_first_token_latency.png",
    },
    {
        "column": "average_inter_token_latency",
        "title": "Average Inter-Token Latency",
        "ylabel": "Latency (seconds)",
        "filename": "inter_token_latency.png",
    },
    {
        "column": "aggregate_token_throughput",
        "title": "Aggregate Token Throughput",
        "ylabel": "Tokens per second",
        "filename": "throughput_comparison.png",
    },
]

for plot in plots:
    values = df[plot["column"]]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    plt.title(plot["title"])
    plt.ylabel(plot["ylabel"])
    plt.xlabel("Execution Strategy")

    # Display the measured value above each bar
    plt.bar_label(bars, fmt="%.3f", padding=4)

    plt.tight_layout()

    output_path = os.path.join("graphs", plot["filename"])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_path}")

    plt.close()

print("\nAll experimental graphs generated successfully.")
