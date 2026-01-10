import os
from celery import Celery

CELERY_BROKER_URL = (
    os.getenv("CELERY_BROKER_URL")
    or os.getenv("REDIS_URL")
    or "redis://localhost:6379/0"
)

CELERY_RESULT_BACKEND = (
    os.getenv("CELERY_RESULT_BACKEND")
    or CELERY_BROKER_URL
)

celery = Celery("vad_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(
    task_serializer="pickle",
    accept_content=["pickle", "json"],
    result_serializer="pickle",
    task_track_started=True,
)

import app.tasks
