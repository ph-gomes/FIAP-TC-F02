"""Logging estruturado e monitoramento de chamadas (latência, sucesso/erro).

Cada chamada monitorada por `monitor_call` grava tanto no logger estruturado
(formato JSON) quanto em `performance_metrics`, para facilitar tracking de
desempenho e envio posterior a ferramentas de observabilidade.
"""

import functools
import json
import logging
import time

logger = logging.getLogger("rf_ga_llm_pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()


class JsonLogFormatter(logging.Formatter):
    """Formata cada linha de log como um objeto JSON, facilitando o envio
    posterior para ferramentas de observabilidade (CloudWatch, Stackdriver, ELK)."""

    def format(self, record):
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, ensure_ascii=False)


_console_handler = logging.StreamHandler()
_console_handler.setFormatter(JsonLogFormatter())

_file_handler = logging.FileHandler("pipeline_metrics.log", encoding="utf-8")
_file_handler.setFormatter(JsonLogFormatter())

logger.addHandler(_console_handler)
logger.addHandler(_file_handler)

# Registro em memória de todas as chamadas monitoradas.
performance_metrics = []


def monitor_call(operation_name):
    """Decorator que mede latência, sucesso/erro de uma operação e
    grava tanto no logger estruturado quanto em `performance_metrics`."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            error_message = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                record = {
                    "operation": operation_name,
                    "duration_ms": round(duration_ms, 2),
                    "status": status,
                    "error": error_message,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                performance_metrics.append(record)
                logger.info(
                    f"{operation_name} finalizado em {duration_ms:.1f}ms ({status})",
                    extra={"extra_fields": record},
                )
        return wrapper
    return decorator
