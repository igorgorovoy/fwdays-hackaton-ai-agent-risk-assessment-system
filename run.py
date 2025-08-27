import os
import logging
from dotenv import load_dotenv

# Відключаємо телеметрію ChromaDB на самому початку
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # виводимо в консоль
        logging.FileHandler('tarot.log')  # зберігаємо в файл
    ]
)

# Load environment variables
load_dotenv()

if __name__ == '__main__':
    port = int(os.getenv('FLASK_RUN_PORT', 5000))
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', '1') == '1'
    
    logging.info(f"Starting Tarot application on {host}:{port}")
    
    app.run(
        debug=debug,
        host=host,
        port=port
    )
