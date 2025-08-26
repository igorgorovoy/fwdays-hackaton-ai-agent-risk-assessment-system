"""
Script to initialize the vector database with tarot card data
"""
import os
import logging
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Load environment variables
        logger.info("Завантаження змінних середовища...")
        load_dotenv()
        
        # Виведення важливих змінних середовища (без секретів)
        logger.info("Перевірка конфігурації:")
        logger.info(f"CARDS_DATA_PATH: {os.getenv('CARDS_DATA_PATH', './app/static/images/cards')}")
        logger.info(f"VECTOR_STORE_PATH: {os.getenv('VECTOR_STORE_PATH', './vector_store')}")
        logger.info(f"MODEL_NAME: {os.getenv('MODEL_NAME', 'gpt-4-turbo-preview')}")
        logger.info(f"OPENAI_API_KEY налаштовано: {'Так' if os.getenv('OPENAI_API_KEY') else 'Ні'}")
        
        logger.info("Ініціалізація Tarot Agent...")
        
        # Initialize agent
        agent = TarotAgent(
            cards_path=os.getenv('CARDS_DATA_PATH', './app/static/images/cards'),
            vector_store_path=os.getenv('VECTOR_STORE_PATH', './vector_store')
        )
        
        logger.info("Завантаження даних карт та створення векторного сховища...")
        # Initialize vector store
        agent.initialize_vector_store()
        
        logger.info("Ініціалізація векторного сховища завершена успішно!")
        logger.info(f"Дані збережено в: {os.getenv('VECTOR_STORE_PATH', './vector_store')}")
        
    except Exception as e:
        logger.error(f"Помилка при ініціалізації: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
