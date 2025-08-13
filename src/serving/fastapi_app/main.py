# src/serving/fastapi_app/main.py

import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np
import torchvision.transforms as T
import mlflow
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from time import time

APP = FastAPI(title="HJ Image Classification API")

# Metrics
REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["path","method","status"])
REQ_LAT = Histogram("http_request_duration_seconds", "Request latency (s)", ["path","method"])
REQ_INPROG = Gauge("http_requests_in_progress", "Requests in progress")
PRED_CLASS = Counter("predicted_class_total", "Predicted class counter", ["class_id"])

@APP.middleware("http")
async def metrics_middleware(request: Request, call_next):
    REQ_INPROG.inc()
    start = time()
    try:
        response = await call_next(request)
        status = getattr(response, "status_code", 500)
        REQ_COUNT.labels(request.url.path, request.method, str(status)).inc()
        REQ_LAT.labels(request.url.path, request.method).observe(time()-start)
        return response
    except Exception:
        REQ_COUNT.labels(request.url.path, request.method, "500").inc()
        REQ_LAT.labels(request.url.path, request.method).observe(time()-start)
        raise
    finally:
        REQ_INPROG.dec()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MODEL_NAME","imgcls-resnet")
DEFAULT_STAGE = os.getenv("MODEL_STAGE","Production")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# CIFAR-10 전처리
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
transform = T.Compose([
    T.Resize((32,32)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

model = None

def load_model_from_registry(model_name: str, stage_or_version: Optional[str]) -> mlflow.pyfunc.PyFuncModel:
    if stage_or_version and stage_or_version.isdigit():
        uri = f"models:/{model_name}/{stage_or_version}"
    else:
        uri = f"models:/{model_name}/{stage_or_version or 'Production'}"
    return mlflow.pyfunc.load_model(uri)

@APP.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model_from_registry(MODEL_NAME, DEFAULT_STAGE)
    except Exception:
        infos = mlflow.client.MlflowClient().get_latest_versions(MODEL_NAME)
        if not infos:
            raise RuntimeError("No model versions found in registry")
        model = load_model_from_registry(MODEL_NAME, infos[-1].version)

@APP.get("/health")
def health():
    return {"status":"ok", "model_loaded": model is not None}

@APP.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

class PredictOut(BaseModel):
    pred:int

@APP.post("/predict", response_model=PredictOut)
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error":"model not loaded"})
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0)
    y = model.predict(x.numpy())
    pred = int(np.argmax(y, axis=1)[0])
    PRED_CLASS.labels(str(pred)).inc()
    return {"pred": pred}
