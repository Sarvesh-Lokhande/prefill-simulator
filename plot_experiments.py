import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)

df = pd.read_csv("results/chunk_size_experiments.csv")

# The last configuration is normal prefill, not chunked prefill.
df["configuration"] = df["tested_chunk_size"].apply(
    lambda size: "Normal Prefill" if size == 4000
    else f"Chunk Size {int(size)}"
)

# 1. New prompt TTFT versus chunk size
plt.figure(figsize=(9, 5))
plt.plot(
    df["tested_chunk_size"],
    df["new_prompt_ttft"],
    marker="o",
)
plt.xlabel("Chunk Size (tokens)")
plt.ylabel("Estimated New Prompt TTFT (seconds)")
plt.title("Effect of Chunk Size on New Prompt TTFT")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/chunk_size_ttft.png", dpi=300)
plt.close()

# 2. Existing users' first-token latency versus chunk size
plt.figure(figsize=(9, 5))
plt.plot(
    df["tested_chunk_size"],
    df["average_decode_first_token_latency"],
    marker="o",
)
plt.xlabel("Chunk Size (tokens)")
plt.ylabel("Average First-Token Latency (seconds)")
plt.title("Effect of Chunk Size on Existing Decode Users")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/chunk_size_decode_latency.png", dpi=300)
plt.close()

# 3. Inter-token latency versus chunk size
plt.figure(figsize=(9, 5))
plt.plot(
    df["tested_chunk_size"],
    df["average_inter_token_latency"],
    marker="o",
)
plt.xlabel("Chunk Size (tokens)")
plt.ylabel("Average Inter-Token Latency (seconds)")
plt.title("Effect of Chunk Size on Inter-Token Latency")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/chunk_size_inter_token_latency.png", dpi=300)
plt.close()

# 4. Throughput versus chunk size
plt.figure(figsize=(9, 5))
plt.plot(
    df["tested_chunk_size"],
    df["aggregate_token_throughput"],
    marker="o",
)
plt.xlabel("Chunk Size (tokens)")
plt.ylabel("Aggregate Throughput (tokens/second)")
plt.title("Effect of Chunk Size on Aggregate Throughput")
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/chunk_size_throughput.png", dpi=300)
plt.close()

print("Generated chunk-size experiment graphs:")
for filename in [
    "chunk_size_ttft.png",
    "chunk_size_decode_latency.png",
    "chunk_size_inter_token_latency.png",
    "chunk_size_throughput.png",
]:
    print(f"graphs/{filename}")
