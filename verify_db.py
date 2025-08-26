"""
Script to verify the vector database content and functionality
"""
import os
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent
from app.tarot_agent.data_loader import TarotDataLoader

def verify_cards_count(agent):
    """Перевіряє кількість карт в базі даних"""
    # Очікувана кількість карт
    expected_major = 22  # Старші аркани (0-21)
    expected_minor = 56  # Молодші аркани (14 карт * 4 масті)
    expected_total = expected_major + expected_minor
    
    # Завантажуємо всі карти через DataLoader для порівняння
    loader = TarotDataLoader(agent.cards_path)
    cards = loader.load_all_cards()
    
    print("\n=== Перевірка кількості карт ===")
    print(f"Знайдено карт: {len(cards)}")
    print(f"Очікувана кількість: {expected_total}")
    print(f"Старші аркани: {sum(1 for card in cards if card.card_type == 'major')}/{expected_major}")
    print(f"Молодші аркани: {sum(1 for card in cards if card.card_type == 'minor')}/{expected_minor}")
    
    return len(cards) == expected_total

def verify_search_functionality(agent):
    """Перевіряє функціональність пошуку"""
    print("\n=== Перевірка пошуку ===")
    
    # Тестові запити для перевірки
    test_queries = [
        "The Fool",
        "Ace of Cups",
        "Ten of Pentacles",
        "The High Priestess"
    ]
    
    for query in test_queries:
        print(f"\nПошук карти: {query}")
        try:
            result = agent.get_card_info(query)
            if result:
                print("✓ Знайдено інформацію")
                # Перевіряємо наявність всіх полів метаданих
                metadata = result['metadata']
                print(f"Тип карти: {metadata['type']}")
                print(f"Масть: {metadata['suit']}")
            else:
                print("✗ Карту не знайдено!")
        except Exception as e:
            print(f"✗ Помилка при пошуку: {str(e)}")

def verify_content_quality(agent):
    """Перевіряє якість збереженого контенту"""
    print("\n=== Перевірка якості контенту ===")
    
    # Перевіряємо The Fool як приклад Старших Арканів
    fool_info = agent.get_card_info("The Fool")
    if fool_info:
        content = fool_info['content']
        print("\nПеревірка The Fool:")
        print("✓ Назва карти присутня" if "Fool" in content else "✗ Назва карти відсутня")
        print("✓ Опис присутній" if len(content) > 100 else "✗ Опис закороткий")
        print("✓ Метадані коректні" if fool_info['metadata']['type'] == 'major' else "✗ Некоректний тип карти")
        print("✓ Метадані масті коректні" if fool_info['metadata']['suit'] == 'NA' else "✗ Некоректна масть")
    
    # Перевіряємо Ace of Cups як приклад Молодших Арканів
    ace_info = agent.get_card_info("Ace of Cups")
    if ace_info:
        content = ace_info['content']
        print("\nПеревірка Ace of Cups:")
        print("✓ Назва карти присутня" if "Ace of Cups" in content else "✗ Назва карти відсутня")
        print("✓ Опис присутній" if len(content) > 100 else "✗ Опис закороткий")
        print("✓ Метадані коректні" if ace_info['metadata']['type'] == 'minor' else "✗ Некоректний тип карти")
        print("✓ Метадані масті коректні" if ace_info['metadata']['suit'] == 'Cups' else "✗ Некоректна масть")

def main():
    # Load environment variables
    load_dotenv()
    
    print("Починаємо перевірку векторної бази даних...")
    
    # Initialize agent
    agent = TarotAgent(
        cards_path=os.getenv('CARDS_DATA_PATH', './app/static/images/cards'),
        vector_store_path=os.getenv('VECTOR_STORE_PATH', './vector_store')
    )
    
    # Ініціалізуємо векторне сховище
    print("\nІніціалізація векторного сховища...")
    agent.initialize_vector_store()
    print("Ініціалізація завершена\n")
    
    # Проводимо всі перевірки
    cards_ok = verify_cards_count(agent)
    verify_search_functionality(agent)
    verify_content_quality(agent)
    
    print("\n=== Підсумок перевірки ===")
    if cards_ok:
        print("✓ Всі карти успішно завантажені")
    else:
        print("✗ Кількість завантажених карт не відповідає очікуваній!")
    
    print("\nПеревірка завершена!")

if __name__ == "__main__":
    main()
