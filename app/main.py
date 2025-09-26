from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .celery_app import celery
from .tasks import anxiety_task
from celery.result import AsyncResult
import logging

app = FastAPI(title="VAD API", version="1.0.0")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringe en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@app.get("/")
def root():
    return {"status": "ok", "message": "API VAD + Celery activo"}

@app.post("/anxiety_async")
async def anxiety_async(file: UploadFile = File(...), user_id: str = Form(...) ):
    audio_bytes = await file.read()
    task = anxiety_task.delay(audio_bytes, user_id)
    log.info(f"Tarea encolada: {task.id} para user_id={user_id}")
    return {"task_id": task.id, "user_id": user_id}

@app.get("/result/{task_id}")
def get_result(task_id: str):
    res = AsyncResult(task_id, app=celery)
    if not res.ready():
        return {"status": "pending"}
    try:
        result = res.get(timeout=5)
        return {"status": "done", "result": result}
    except Exception as e:
        log.exception("Error al obtener resultado de Celery")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
