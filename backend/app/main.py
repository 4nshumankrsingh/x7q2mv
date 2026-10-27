from fastapi import FastAPI

app = FastAPI(title="Wind Forecast Decision Support", version="0.1.0")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}