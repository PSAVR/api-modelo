from .celery_app import celery
from .model_inference import run_anxiety_analysis
import logging

log = logging.getLogger(__name__)

@celery.task(name="anxiety_task")
def anxiety_task(audio_bytes: bytes, user_id: str, decimals: int = 2):
    """Tarea asíncrona que analiza el audio y devuelve ansiedad + user_id"""
    log.info("=== [anxiety_task] Iniciando análisis ===")
    try:
        log.info(f"Analizando audio para user_id={user_id}")

        # Ejecutar el modelo
        result = run_anxiety_analysis(audio_bytes, decimals)

        # Agregar el user_id al resultado
        result["user_id"] = user_id

        log.info(f"=== [anxiety_task] Completado para user_id={user_id}: {result} ===")
        return result

    except Exception as e:
        log.exception(f"[anxiety_task] Error durante ejecución para user_id={user_id}")
        raise
