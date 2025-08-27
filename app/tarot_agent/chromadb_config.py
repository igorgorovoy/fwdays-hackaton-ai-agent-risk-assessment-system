"""
Глобальна конфігурація для ChromaDB
"""
import os
import chromadb
from chromadb.config import Settings

def get_chromadb_settings() -> Settings:
    """
    Повертає стандартні налаштування ChromaDB з відключеною телеметрією
    """
    return Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )

def configure_chromadb_globally():
    """
    Налаштовує ChromaDB глобально для всього проекту
    """
    # Відключаємо телеметрію через змінні середовища
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_SERVER_NOFILE"] = "1"
    
    # Блокуємо телеметрію на рівні ChromaDB
    try:
        import chromadb.telemetry.product.posthog
        # Замінюємо метод capture на пустий
        chromadb.telemetry.product.posthog.Posthog.capture = lambda *args, **kwargs: None
        
        # Також блокуємо основний клас телеметрії
        import chromadb.telemetry.product
        chromadb.telemetry.product.Telemetry._submit_event = lambda *args, **kwargs: None
        chromadb.telemetry.product.Telemetry.capture = lambda *args, **kwargs: None
    except (ImportError, AttributeError):
        pass
    
    # Додатково блокуємо на рівні логування
    try:
        import logging
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
    except:
        pass
    
    return get_chromadb_settings()
