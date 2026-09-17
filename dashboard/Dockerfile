FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./dashboard/
COPY data/raw/get_around_delay_analysis.xlsx ./data/raw/get_around_delay_analysis.xlsx

EXPOSE 7860

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]