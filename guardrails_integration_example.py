"""
Приклад інтеграції Guardrails в Tarot AI агент
Демонструє як додати захисні механізми до існуючого коду
"""

import asyncio
import os
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent
from app.tarot_agent.guardrails import TarotGuardrails, apply_guardrails, GuardrailsContext

class TarotAgentWithGuardrails(TarotAgent):
    """Розширений Tarot Agent з інтегрованими Guardrails"""
    
    def __init__(self, cards_path: str, vector_store_path: str):
        super().__init__(cards_path, vector_store_path)
        
        # Ініціалізуємо Guardrails
        self.guardrails = TarotGuardrails()
        self.logger.info("Guardrails ініціалізовано успішно")
    
    async def get_reading_safe(self, question: str, num_cards: int = 3) -> dict:
        """
        Безпечний метод отримання читання з Guardrails
        
        Args:
            question: Питання користувача
            num_cards: Кількість карт для витягування
            
        Returns:
            dict: Результат читання або помилка
            
        Raises:
            ValueError: Якщо Guardrails заблокували запит
        """
        # Використовуємо контекстний менеджер для моніторингу
        with GuardrailsContext(self.guardrails) as guardrails:
            
            # 1. Перевірка вхідних даних
            input_result = guardrails.input_validation(question)
            if not input_result.is_safe:
                self.logger.warning(f"Input заблоковано: {input_result.reason}")
                raise ValueError(f"Недійсний запит: {input_result.reason}")
            
            # 2. Перевірка відповідності контенту
            content_result = guardrails.content_appropriateness(question)
            if not content_result.is_safe:
                self.logger.warning(f"Content заблоковано: {content_result.reason}")
                raise ValueError(f"Неприйнятний контент: {content_result.reason}")
            
            # 3. Перевірка rate limiting
            rate_result = guardrails.rate_limiting()
            if not rate_result.is_safe:
                self.logger.warning(f"Rate limit досягнуто: {rate_result.reason}")
                raise ValueError(f"Перевищено ліміт запитів: {rate_result.reason}")
            
            try:
                # 4. Виконуємо основне читання
                self.logger.info(f"Обробка безпечного запиту: {question}")
                result = await super().get_reading(question, num_cards)
                
                # 5. Перевірка якості витягнутого контексту (якщо доступно)
                # Тут би була перевірка retrieved_docs, але в нашому випадку це внутрішня логіка
                
                # 6. Перевірка вихідної відповіді
                output_result = guardrails.output_validation(
                    result.get('reading', ''), 
                    result.get('cards', [])
                )
                
                if not output_result.is_safe:
                    self.logger.warning(f"Output validation warning: {output_result.reason}")
                    # Не блокуємо, але логуємо для моніторингу
                    result['guardrails_warning'] = output_result.reason
                
                # 7. Додаємо метадані Guardrails
                result['guardrails_metadata'] = {
                    'input_confidence': input_result.confidence,
                    'content_appropriate': content_result.is_safe,
                    'output_quality': output_result.confidence if output_result else 1.0
                }
                
                self.logger.info("Запит успішно оброблено з Guardrails")
                return result
                
            except Exception as e:
                self.logger.error(f"Помилка при обробці запиту: {str(e)}")
                raise

    @apply_guardrails(TarotGuardrails())
    async def get_reading_with_decorator(self, question: str, num_cards: int = 3) -> dict:
        """
        Альтернативний підхід - використання декоратора
        Guardrails застосовуються автоматично
        """
        return await super().get_reading(question, num_cards)

    def get_guardrails_report(self) -> dict:
        """Отримання детального звіту про роботу Guardrails"""
        stats = self.guardrails.get_stats()
        
        return {
            "summary": {
                "total_requests": stats["total_requests"],
                "blocked_requests": stats["blocked_requests"],
                "success_rate": (1 - stats["block_rate"]) * 100,
                "average_processing_time": stats["average_processing_time"]
            },
            "detailed_stats": stats,
            "status": "healthy" if stats["block_rate"] < 0.1 else "attention_needed"
        }

async def demo_guardrails():
    """Демонстрація роботи Guardrails"""
    
    print("🛡️ Демонстрація роботи Guardrails для Tarot AI")
    print("=" * 60)
    
    # Ініціалізація агента з Guardrails
    load_dotenv()
    agent = TarotAgentWithGuardrails(
        cards_path="./app/static/images/cards",
        vector_store_path="./vector_store"
    )
    agent.initialize_vector_store()
    
    # Тестові запити
    test_cases = [
        {
            "question": "Що означає карта Маг у контексті кар'єри?",
            "expected": "success",
            "description": "Нормальний запит про Таро"
        },
        {
            "question": "Скажи мені діагноз хвороби за картами",
            "expected": "blocked",
            "description": "Медична порада (заборонено)"
        },
        {
            "question": "aaaaaaaaaaaaaaaaaaaaaa" * 50,
            "expected": "blocked", 
            "description": "Спам (занадто довгий)"
        },
        {
            "question": "Які перспективи мого бізнесу?",
            "expected": "success",
            "description": "Нормальний бізнес-запит"
        },
        {
            "question": "",
            "expected": "blocked",
            "description": "Порожній запит"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: {test_case['description']}")
        print(f"Запит: '{test_case['question'][:50]}{'...' if len(test_case['question']) > 50 else ''}'")
        
        try:
            # Тестуємо з Guardrails
            result = await agent.get_reading_safe(test_case["question"])
            
            print("✅ Запит успішно оброблено")
            print(f"Карти: {[card['name'] for card in result.get('cards', [])]}")
            
            if 'guardrails_warning' in result:
                print(f"⚠️ Попередження: {result['guardrails_warning']}")
            
            results.append({"test": i, "status": "success", "result": result})
            
        except ValueError as e:
            print(f"❌ Запит заблоковано: {str(e)}")
            results.append({"test": i, "status": "blocked", "reason": str(e)})
            
        except Exception as e:
            print(f"💥 Несподівана помилка: {str(e)}")
            results.append({"test": i, "status": "error", "error": str(e)})
    
    # Звіт про результати
    print("\n" + "=" * 60)
    print("📊 ЗВІТ ПРО ТЕСТУВАННЯ")
    print("=" * 60)
    
    report = agent.get_guardrails_report()
    print(f"Всього запитів: {report['summary']['total_requests']}")
    print(f"Заблоковано: {report['summary']['blocked_requests']}")
    print(f"Рівень успіху: {report['summary']['success_rate']:.1f}%")
    print(f"Статус системи: {report['status']}")
    
    # Детальна статистика
    if report['detailed_stats']['blocked_reasons']:
        print(f"\nПричини блокування:")
        for reason, count in report['detailed_stats']['blocked_reasons'].items():
            print(f"  • {reason}: {count}")
    
    return results

async def test_performance_with_guardrails():
    """Тест продуктивності з Guardrails"""
    
    print("\n🚀 Тест продуктивності Guardrails")
    print("=" * 40)
    
    load_dotenv()
    agent = TarotAgentWithGuardrails(
        cards_path="./app/static/images/cards",
        vector_store_path="./vector_store"
    )
    agent.initialize_vector_store()
    
    # Тестові запити
    questions = [
        "Що мене чекає в кар'єрі?",
        "Яки перспективи стосунків?",
        "Як покращити фінансове становище?",
        "Що радять карти щодо здоров'я?",
        "Яке майбутнє мого проекту?"
    ]
    
    import time
    
    # Тест без Guardrails (стандартний метод)
    start_time = time.time()
    for question in questions:
        try:
            await agent.get_reading(question)
        except:
            pass
    no_guardrails_time = time.time() - start_time
    
    # Скидаємо статистику
    agent.guardrails.reset_stats()
    
    # Тест з Guardrails
    start_time = time.time()
    for question in questions:
        try:
            await agent.get_reading_safe(question)
        except:
            pass
    with_guardrails_time = time.time() - start_time
    
    print(f"Без Guardrails: {no_guardrails_time:.2f}s")
    print(f"З Guardrails: {with_guardrails_time:.2f}s")
    print(f"Overhead: {((with_guardrails_time - no_guardrails_time) / no_guardrails_time * 100):.1f}%")

if __name__ == "__main__":
    # Запуск демонстрації
    asyncio.run(demo_guardrails())
    
    # Запуск тесту продуктивності
    asyncio.run(test_performance_with_guardrails())
