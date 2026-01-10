import os
import ssl
from celery import Celery

def _redis_url():
    return (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("CELERY_RESULT_BACKEND")
        or os.getenv("REDIS_URL")
        or os.getenv("HEROKU_REDIS_YELLOW_URL")
        or "redis://localhost:6379/0"
    )

CELERY_BROKER_URL = _redis_url()
CELERY_RESULT_BACKEND = _redis_url()

celery = Celery("vad_tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


if CELERY_BROKER_URL.startswith("rediss://"):
    ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE}
    celery.conf.broker_use_ssl = ssl_opts
    celery.conf.redis_backend_use_ssl = ssl_opts

celery.conf.update(
    task_serializer="pickle",
    accept_content=["pickle", "json"],
    result_serializer="pickle",
    task_track_started=True,
)

import app.tasks
