# Chunked Prefill Simulator

## 1. Overview

The Chunked Prefill Simulator is a Python-based simulation project that
compares normal prefill and chunked prefill execution strategies.

The project investigates how dividing a prompt into smaller chunks can
affect time to first token (TTFT), decode latency, inter-token latency,
and token throughput.

A Streamlit dashboard provides an interactive interface for configuring
experiments and visualizing results. Docker and Docker Compose are used
to package and run the application.

This project uses a simplified discrete-event simulation. It does not
execute a real large language model or measure actual GPU performance.

## 2. Objectives

-   Implement normal prefill and chunked prefill execution models.
-   Compare TTFT and decode latency under configurable assumptions.
-   Study the effect of chunk size on simulated performance.
-   Generate experimental graphs.
-   Provide an interactive dashboard.
-   Package the application using Docker.
-   Support reproducible experiments.

## 3. Technologies Used

-   Python
-   SimPy
-   Pandas
-   Matplotlib
-   Streamlit
-   Docker
-   Docker Compose

## 4. Project Structure

``` text
prefill-simulator/
├── app.py
├── simulator.py
├── experiment.py
├── plot_results.py
├── plot_experiments.py
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── README.md
├── results/
└── graphs/
```

## 5. Prefill Strategies

### Normal Prefill

Normal prefill processes the entire input prompt before moving to the
next modeled stage.

For a prompt containing P tokens and a configured processing rate R, the
modeled prefill duration is:

`Prefill Duration = P / R`

### Chunked Prefill

Chunked prefill divides the input prompt into smaller chunks. Each chunk
is processed sequentially, and the simulation can schedule decode
operations between chunks according to its scheduling policy.

If the prompt contains P tokens and the chunk size is C, the approximate
number of chunks is:

`Number of Chunks = Ceiling(P / C)`

The simulator also includes a fixed overhead per chunk.

## 6. Performance Metrics

### Time to First Token (TTFT)

TTFT measures the simulated elapsed time until the first output token
for the new prompt is produced.

### Decode First-Token Latency

Measures the elapsed time until an existing decode user's first output
token is produced.

### Inter-Token Latency

Measures the average interval between successive generated tokens for
existing decode users.

### Aggregate Token Throughput

The simulator calculates aggregate token throughput as:

`Aggregate Throughput = Total Modeled Tokens / Simulation Time`

This metric includes modeled prompt-processing tokens and output tokens.
It should not be interpreted as actual GPU or production LLM throughput.

### Output Token Throughput

Output token throughput measures the number of modeled output tokens
divided by simulated execution time.

## 7. Installation and Local Execution

### Prerequisites

-   Python 3.10 or later
-   pip

### Create a Virtual Environment

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

``` bash
pip install -r requirements.txt
```

### Run the Simulator

``` bash
python simulator.py
```

### Run Experiments

``` bash
python experiment.py
```

### Generate Graphs

``` bash
python plot_results.py
python plot_experiments.py
```

### Launch the Dashboard

``` bash
streamlit run app.py
```

Open the URL displayed by Streamlit, usually:

`http://localhost:8501`

## 8. Docker Deployment

Docker packages the simulator and its dependencies into a container.
Docker Compose simplifies building and running the application.

### Build and Start

``` bash
sudo docker compose up -d --build
```

### Check Container Status

``` bash
sudo docker compose ps
```

### View Logs

``` bash
sudo docker compose logs -f
```

### Open the Dashboard

`http://localhost:8501`

### Stop the Application

``` bash
sudo docker compose down
```

The Compose configuration mounts the local `results/` and `graphs/`
directories inside the container.

## 9. Experimental Evaluation

The simulator supports experiments using different chunk sizes.

Example configuration:

-   Prompt length: 4000 tokens
-   Processing rate: 1000 tokens per second
-   Existing decode users: 2
-   Output tokens per existing user: 20
-   Decode time per token: 0.05 seconds
-   Chunk overhead: 0.005 seconds per chunk

Example chunk sizes:

`250, 500, 1000, 2000, 4000`

The experiment scripts generate CSV results and graphs for comparing
latency and throughput.

The numerical results depend on the simulator's execution rules and
configured assumptions. They are not real hardware benchmarks.

Run the experiment scripts again to reproduce the measurements after
confirming that the simulator configuration has not changed.

## 10. Limitations

-   The simulator does not run an actual language model.
-   GPU execution, memory consumption, and hardware contention are not
    measured.
-   The current implementation uses simplified sequential scheduling.
-   Results depend on the configured processing rate, decode time, chunk
    overhead, and scheduling policy.
-   The simulator is intended to study modeled behavior, not to
    reproduce production inference performance.

## 11. Reference

SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked
Prefills.

Paper: https://arxiv.org/abs/2308.16369

The paper provides background on chunked prefill and scheduling in large
language model inference. This simulator is a simplified educational
implementation and is not a full reproduction of SARATHI.
