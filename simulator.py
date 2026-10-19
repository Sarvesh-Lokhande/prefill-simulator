import simpy
import pandas as pd
import os

# =========================================================
# DEFAULT CONFIGURATION
# =========================================================

PROCESSING_RATE = 1000
CHUNK_SIZE = 500
DECODE_TIME = 0.05
DECODE_USERS = 2
OUTPUT_TOKENS = 20

os.makedirs("results", exist_ok=True)


# =========================================================
# SIMULATION ENGINE
# =========================================================

def run_experiment(
    mode,
    prompt_length=4000,
    chunk_size=CHUNK_SIZE,
    processing_rate=PROCESSING_RATE,
    decode_users=DECODE_USERS,
    output_tokens=OUTPUT_TOKENS,
    decode_time=DECODE_TIME,
    chunk_overhead=0.005,
):
    """
    Simplified discrete-event model of normal and chunked prefill.

    Assumptions:
    - One shared, single-capacity processing resource.
    - One new prompt arrives at time zero.
    - Existing decode requests are ready at time zero.
    - Prefill duration is proportional to prompt length.
    - Decode operations have fixed duration.
    - Normal prefill is one non-preemptive operation.
    - Chunked prefill releases the resource after each chunk.
    - Existing decode requests are offered service between chunks.
    - This is a scheduling simulation, not a GPU benchmark.
    """

    if mode not in ("normal", "chunked"):
        raise ValueError("mode must be 'normal' or 'chunked'")

    if prompt_length <= 0 or processing_rate <= 0:
        raise ValueError("Prompt length and processing rate must be positive")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if decode_users < 0 or output_tokens < 0:
        raise ValueError("Decode users and output tokens cannot be negative")

    if decode_time <= 0:
        raise ValueError("Decode time must be positive")

    if chunk_overhead < 0:
        raise ValueError("Chunk overhead cannot be negative")

    env = simpy.Environment()
    compute = simpy.Resource(env, capacity=1)

    events = []
    token_times = {"NewPrompt": []}

    active_users = [
        f"DecodeUser{i + 1}" for i in range(decode_users)
    ]

    for user in active_users:
        token_times[user] = []

    generated = {user: 0 for user in active_users}

    prefill_completion = None
    new_prompt_first_token_time = None

    def execute_operation(
        event_name,
        user_id,
        duration,
        token_index=None,
        tokens_processed=0,
    ):
        """Acquire the shared resource and execute one operation."""

        with compute.request() as request:
            yield request

            start_time = env.now
            yield env.timeout(duration)
            finish_time = env.now

        events.append({
            "start_time": start_time,
            "time": finish_time,
            "duration": duration,
            "event": event_name,
            "user_id": user_id,
            "token_index": token_index,
            "tokens_processed": tokens_processed,
        })

        if event_name == "decode_token":
            token_times[user_id].append(finish_time)

    def decode_one_token(user_id):
        """Generate and record one output token."""

        token_index = len(token_times[user_id]) + 1

        yield from execute_operation(
            event_name="decode_token",
            user_id=user_id,
            duration=decode_time,
            token_index=token_index,
            tokens_processed=1,
        )

        if user_id in generated:
            generated[user_id] += 1

    def decode_round():
        """Give each existing decode user one token opportunity."""

        for user in active_users:
            if generated[user] < output_tokens:
                yield from decode_one_token(user)

    def experiment():
        nonlocal prefill_completion
        nonlocal new_prompt_first_token_time

        events.append({
            "start_time": 0.0,
            "time": 0.0,
            "duration": 0.0,
            "event": "request_arrival",
            "user_id": "NewPrompt",
            "token_index": None,
            "tokens_processed": 0,
        })

        if mode == "normal":

            # A single long, non-preemptive prefill operation.
            yield from execute_operation(
                event_name="prefill",
                user_id="NewPrompt",
                duration=prompt_length / processing_rate,
                tokens_processed=prompt_length,
            )

            prefill_completion = env.now

        else:

            # The prompt is divided into independently scheduled chunks.
            remaining = prompt_length

            while remaining > 0:
                current_chunk = min(chunk_size, remaining)

                yield from execute_operation(
                    event_name="prefill_chunk",
                    user_id="NewPrompt",
                    duration=(
                        current_chunk / processing_rate
                        + chunk_overhead
                    ),
                    tokens_processed=current_chunk,
                )

                remaining -= current_chunk

                # Let existing requests compete between chunks.
                if remaining > 0:
                    yield from decode_round()

            prefill_completion = env.now

        events.append({
            "start_time": env.now,
            "time": env.now,
            "duration": 0.0,
            "event": "prefill_complete",
            "user_id": "NewPrompt",
            "token_index": None,
            "tokens_processed": 0,
        })

        # Generate the new request's first output token.
        yield from decode_one_token("NewPrompt")
        new_prompt_first_token_time = env.now

        # Complete the remaining output tokens of existing users.
        while any(
            generated[user] < output_tokens
            for user in active_users
        ):
            yield from decode_round()

    env.process(experiment())
    env.run()

    # =====================================================
    # METRICS
    # =====================================================

    decode_metrics = []

    for user in active_users:
        times = token_times[user]

        inter_token_latencies = [
            times[i] - times[i - 1]
            for i in range(1, len(times))
        ]

        decode_metrics.append({
            "user_id": user,
            "first_decode_token_latency": (
                times[0] if times else 0.0
            ),
            "average_inter_token_latency": (
                sum(inter_token_latencies) / len(inter_token_latencies)
                if inter_token_latencies else 0.0
            ),
            "generated_tokens": len(times),
        })

    decode_df = pd.DataFrame(
        decode_metrics,
        columns=[
            "user_id",
            "first_decode_token_latency",
            "average_inter_token_latency",
            "generated_tokens",
        ],
    )

    total_decode_tokens = sum(
        len(token_times[user]) for user in active_users
    )

    new_prompt_output_tokens = len(token_times["NewPrompt"])

    total_output_tokens = total_decode_tokens + new_prompt_output_tokens

    total_tokens = prompt_length + total_output_tokens

    simulation_time = env.now

    summary = {
        "mode": mode,
        "prompt_length": prompt_length,
        "chunk_size": chunk_size if mode == "chunked" else prompt_length,
        "prefill_completion_time": prefill_completion,
        "new_prompt_ttft": new_prompt_first_token_time,
        "average_decode_first_token_latency": (
            decode_df["first_decode_token_latency"].mean()
            if not decode_df.empty else 0.0
        ),
        "average_inter_token_latency": (
            decode_df["average_inter_token_latency"].mean()
            if not decode_df.empty else 0.0
        ),
        "total_decode_tokens": total_decode_tokens,
        "new_prompt_output_tokens": new_prompt_output_tokens,
        "simulation_time": simulation_time,
        "aggregate_token_throughput": (
            total_tokens / simulation_time if simulation_time > 0 else 0.0
        ),
        "output_token_throughput": (
            total_output_tokens / simulation_time
            if simulation_time > 0 else 0.0
        ),
    }

    return summary, events, decode_df


# =========================================================
# RUN BOTH EXPERIMENTS
# =========================================================

if __name__ == "__main__":

    normal, normal_events, normal_decode = run_experiment(
        mode="normal"
    )

    chunked, chunked_events, chunked_decode = run_experiment(
        mode="chunked"
    )

    comparison = pd.DataFrame([normal, chunked])

    print("\nPREFILL AND DECODE COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))

    print("\nNORMAL PREFILL: DECODE METRICS")
    print("=" * 100)
    print(normal_decode.to_string(index=False))

    print("\nCHUNKED PREFILL: DECODE METRICS")
    print("=" * 100)
    print(chunked_decode.to_string(index=False))

    comparison.to_csv("results/decode_comparison.csv", index=False)

    pd.DataFrame(normal_events).to_csv(
        "results/normal_events.csv", index=False
    )

    pd.DataFrame(chunked_events).to_csv(
        "results/chunked_events.csv", index=False
    )

    normal_decode.to_csv(
        "results/normal_decode_metrics.csv", index=False
    )

    chunked_decode.to_csv(
        "results/chunked_decode_metrics.csv", index=False
    )

    print("\nAll results saved in the results/ directory.")
