import os
from flask import Flask

# Відключаємо телеметрію ChromaDB на самому початку
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Налаштовуємо ChromaDB глобально перед імпортом інших модулів
try:
    from app.tarot_agent.chromadb_config import configure_chromadb_globally
    configure_chromadb_globally()
except ImportError:
    # Якщо модуль не доступний, продовжуємо без конфігурації
    pass

app = Flask(__name__)

from app import routes
