FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulator.py .
COPY experiment.py .
COPY plot_results.py .
COPY plot_experiments.py .
COPY app.py .

RUN mkdir -p results graphs

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
