#!/usr/bin/env bash
set -e
exec gunicorn app.main:app \
  --workers ${WORKERS:-3} \
  --threads ${THREADS:-1} \
  --timeout ${TIMEOUT:-120} \
  --bind 0.0.0.0:8080 \
  -k uvicorn.workers.UvicornWorker
