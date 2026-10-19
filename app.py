import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from simulator import run_experiment

st.set_page_config(
    page_title="Chunked Prefill Simulator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Chunked Prefill Simulator")
st.markdown(
    """
    Compare **normal prefill** and **chunked prefill** using a discrete-event
    simulation. Explore how chunk size affects latency and throughput.
    """
)

st.info(
    "These are simulated results based on configurable timing assumptions. "
    "This dashboard does not execute a real large language model or measure GPU performance."
)

# --------------------------------------------------
# SIDEBAR: EXPERIMENT CONFIGURATION
# --------------------------------------------------

st.sidebar.header("Experiment Configuration")

prompt_length = st.sidebar.number_input(
    "Prompt length (tokens)",
    min_value=500,
    max_value=20000,
    value=4000,
    step=500
)

chunk_size = st.sidebar.selectbox(
    "Chunk size (tokens)",
    options=[250, 500, 1000, 2000, 4000],
    index=1
)

processing_rate = st.sidebar.number_input(
    "Processing rate (tokens/second)",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100
)

decode_users = st.sidebar.number_input(
    "Existing decode users",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

output_tokens = st.sidebar.number_input(
    "Output tokens per existing user",
    min_value=1,
    max_value=200,
    value=20,
    step=1
)

decode_time = st.sidebar.number_input(
    "Decode time per token (seconds)",
    min_value=0.001,
    max_value=1.0,
    value=0.05,
    step=0.01,
    format="%.3f"
)

st.sidebar.caption(
    "Chunk overhead is currently fixed by the simulator at 0.005 seconds per chunk."
)

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def run_model(mode, selected_chunk_size):
    """Run one simulation and return its metrics dictionary."""
    metrics, events, decode_df = run_experiment(
        mode=mode,
        prompt_length=int(prompt_length),
        chunk_size=int(selected_chunk_size),
        processing_rate=float(processing_rate),
        decode_users=int(decode_users),
        output_tokens=int(output_tokens),
        decode_time=float(decode_time)
    )

    return metrics

def metrics_to_dataframe(results):
    """Convert simulation results into a comparison table."""
    rows = []

    for mode, result in results.items():
        rows.append({
            "Mode": mode.capitalize(),
            "Prompt length": int(prompt_length),
            "Chunk size": (
                int(chunk_size) if mode == "chunked"
                else int(prompt_length)
            ),
            "Prefill completion (s)": result["prefill_completion_time"],
            "New prompt TTFT (s)": result["new_prompt_ttft"],
            "Decode first-token latency (s)": (
                result["average_decode_first_token_latency"]
            ),
            "Average inter-token latency (s)": (
                result["average_inter_token_latency"]
            ),
            "Aggregate throughput (tokens/s)": (
                result["aggregate_token_throughput"]
            ),
            "Output throughput (tokens/s)": (
                result["output_token_throughput"]
            ),
            "Simulation time (s)": result["simulation_time"]
        })

    return pd.DataFrame(rows)


def display_comparison_chart(df, column, title, ylabel):
    """Display a simple comparison chart."""
    chart_data = df.set_index("Mode")[[column]]

    st.subheader(title)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    chart_data.plot(kind="bar", ax=ax, legend=False)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Prefill mode")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

st.header("Run Experiments")

col1, col2, col3 = st.columns(3)

with col1:
    run_normal_button = st.button(
        "Run Normal Prefill",
        use_container_width=True
    )

with col2:
    run_chunked_button = st.button(
        "Run Chunked Prefill",
        use_container_width=True
    )

with col3:
    compare_button = st.button(
        "Compare Both",
        type="primary",
        use_container_width=True
    )


if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = {}

if run_normal_button:
    with st.spinner("Running normal prefill simulation..."):
        st.session_state.simulation_results = {
            "normal": run_model("normal", int(prompt_length))
        }

if run_chunked_button:
    with st.spinner("Running chunked prefill simulation..."):
        st.session_state.simulation_results = {
            "chunked": run_model("chunked", int(chunk_size))
        }

if compare_button:
    with st.spinner("Running both simulations..."):
        normal_result = run_model("normal", int(prompt_length))
        chunked_result = run_model("chunked", int(chunk_size))

        st.session_state.simulation_results = {
            "normal": normal_result,
            "chunked": chunked_result
        }


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

results = st.session_state.simulation_results

if results:
    st.header("Simulation Results")

    comparison_df = metrics_to_dataframe(results)

    # Metric cards for each mode
    for mode, result in results.items():
        st.subheader(f"{mode.capitalize()} Prefill")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "New Prompt TTFT",
            f'{result["new_prompt_ttft"]:.3f} s'
        )

        c2.metric(
            "Decode First-Token Latency",
            f'{result["average_decode_first_token_latency"]:.3f} s'
        )

        c3.metric(
            "Inter-Token Latency",
            f'{result["average_inter_token_latency"]:.3f} s'
        )

        c4.metric(
            "Aggregate Throughput",
            f'{result["aggregate_token_throughput"]:.2f} tokens/s'
        )

    st.subheader("Numerical Comparison")
    st.dataframe(
        comparison_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Comparison CSV",
        data=comparison_df.to_csv(index=False),
        file_name="prefill_comparison.csv",
        mime="text/csv"
    )

    # Draw comparison charts when both modes are available.
    if len(results) == 2:
        display_comparison_chart(
            comparison_df,
            "New prompt TTFT (s)",
            "Time to First Token (TTFT)",
            "Seconds"
        )

        display_comparison_chart(
            comparison_df,
            "Decode first-token latency (s)",
            "Decode First-Token Latency",
            "Seconds"
        )

        display_comparison_chart(
            comparison_df,
            "Average inter-token latency (s)",
            "Average Inter-Token Latency",
            "Seconds"
        )

        display_comparison_chart(
            comparison_df,
            "Aggregate throughput (tokens/s)",
            "Aggregate Token Throughput",
            "Tokens per second"
        )

    st.caption(
        "Interpretation: smaller chunks may allow existing decode requests "
        "to make progress sooner, while chunk overhead can increase total "
        "prefill time. Results depend on the simulator's scheduling rules "
        "and configured assumptions."
    )

else:
    st.warning(
        "Choose the configuration in the sidebar, then click "
        "'Compare Both' to run your first experiment."
    )


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

with st.expander("How does this simulator work?"):
    st.markdown(
        """
        **Normal prefill:** processes the entire prompt before moving to
        the next modeled stage.

        **Chunked prefill:** divides the prompt into smaller pieces and
        processes those pieces sequentially, allowing decode operations
        between chunks according to the simulator's scheduling policy.

        **TTFT:** time until the modeled first output token is produced
        for the new prompt.

        **Decode first-token latency:** time until an existing decode
        user's first output token is produced.

        **Inter-token latency:** average interval between successive
        output tokens for existing decode users.

        **Aggregate throughput:** total modeled tokens divided by the
        simulated execution time. This is not a measurement of real GPU
        throughput.
        """
    )
