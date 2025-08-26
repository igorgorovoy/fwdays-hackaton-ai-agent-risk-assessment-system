import os
import asyncio
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent

async def check_documents():
    print("Перевірка завантажених документів...")
    
    # Завантажуємо агента
    load_dotenv()
    agent = TarotAgent(
        cards_path="./app/static/images/cards",
        vector_store_path="./vector_store"
    )
    agent.initialize_vector_store()
    
    # Перевіряємо документи для різних карт
    test_cards = [
        # Старші Аркани
        'The Fool',
        'The Magician',
        'Death',
        'The World',
        # Молодші Аркани - по одній з кожної масті
        'Ace of Cups',
        'King of Pentacles',
        'Queen of Swords',
        'Knight of Wands'
    ]
    
    print("\nПеревірка документів для вибраних карт:")
    for card in test_cards:
        print(f"\n=== {card} ===")
        card_info = agent.get_card_info(card)
        if card_info:
            print("Знайдено інформацію:")
            print(f"Контент: {card_info['content'][:200]}...")  # Показуємо перші 200 символів
            print(f"Метадані: {card_info['metadata']}")
        else:
            print("Інформацію не знайдено!")

    # Перевіряємо загальну кількість документів
    docs = agent.vector_store.similarity_search("", k=1000)  # Намагаємось отримати всі документи
    
    print(f"\nЗагальна кількість документів: {len(docs)}")
    
    # Аналізуємо типи документів
    doc_types = {}
    for doc in docs:
        doc_type = doc.metadata.get('type', 'unknown')
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print("\nРозподіл документів за типами:")
    for doc_type, count in doc_types.items():
        print(f"- {doc_type}: {count} документів")

if __name__ == "__main__":
    asyncio.run(check_documents())
