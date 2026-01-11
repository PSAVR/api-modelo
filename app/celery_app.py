import os
import ssl
from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery("vad_tasks", broker=REDIS_URL, backend=REDIS_URL)


if REDIS_URL.startswith("rediss://"):
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
