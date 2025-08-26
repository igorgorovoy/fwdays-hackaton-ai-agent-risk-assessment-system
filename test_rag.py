import os
import asyncio
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent

async def test_vector_search():
    print("Тестуємо пошук у векторній базі...")
    
    # Завантажуємо агента
    load_dotenv()
    agent = TarotAgent(
        cards_path="./app/static/images/cards",
        vector_store_path="./vector_store"
    )
    agent.initialize_vector_store()
    
    # Тестові запити
    test_queries = [
        "Що означає карта Маг?",
        "Розкажи про карту Смерть",
        "Що символізує Колесо Фортуни?",
        "Опиши значення карти Місяць",
        "Що означає Туз Кубків?"
    ]
    
    for query in test_queries:
        print(f"\nЗапит: {query}")
        try:
            # Отримуємо відповідь
            response = await agent.get_reading(query)
            
            # Виводимо результат
            print("\nВитягнуті карти:")
            for card in response["cards"]:
                status = "перевернута" if card["is_reversed"] else "пряма"
                print(f"- {card['name']} ({status})")
            
            print("\nВідповідь:")
            print(response["reading"])
            print("-" * 80)
            
        except Exception as e:
            print(f"Помилка при обробці запиту: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_vector_search())
