import os
import json
import asyncio
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from app.tarot_agent.agent import TarotAgent

class RAGEvaluator:
    def __init__(self):
        """Ініціалізація оцінювача RAG системи"""
        # Тестові кейси
        self.test_cases = [
            {
                "query": "Що означає карта Маг?",
                "expected_cards": ["The Magician"],
                "expected_keywords": ["творчість", "воля", "майстерність", "сила", "талант"],
                "expected_context": ["перша карта старших арканів", "елемент повітря"]
            },
            {
                "query": "Розкажи про карту Смерть",
                "expected_cards": ["Death"],
                "expected_keywords": ["трансформація", "зміни", "завершення", "початок"],
                "expected_context": ["тринадцята карта", "трансформація"]
            },
            {
                "query": "Що символізує Колесо Фортуни?",
                "expected_cards": ["Wheel of Fortune"],
                "expected_keywords": ["доля", "цикл", "зміни", "фортуна"],
                "expected_context": ["десята карта", "карма", "цикли"]
            },
            {
                "query": "Опиши значення карти Місяць",
                "expected_cards": ["The Moon"],
                "expected_keywords": ["інтуїція", "страхи", "ілюзії", "підсвідомість"],
                "expected_context": ["вісімнадцята карта", "нічне світило"]
            },
            {
                "query": "Що означає Туз Кубків?",
                "expected_cards": ["Ace of Cups"],
                "expected_keywords": ["емоції", "любов", "початок", "чаша"],
                "expected_context": ["молодші аркани", "масть кубків"]
            }
        ]
    
    def evaluate_retrieval(self, retrieved_cards: List[str], expected_cards: List[str]) -> Dict[str, float]:
        """
        Оцінка точності витягнутих карт
        
        F1 Score - це гармонічне середнє між Precision і Recall:
        
        Precision (Точність) = Правильно витягнуті карти / Всі витягнуті карти
        - Показує, скільки з витягнутих карт були правильними
        - Приклад: якщо витягли 3 карти і 2 правильні, то Precision = 2/3 = 0.67
        
        Recall (Повнота) = Правильно витягнуті карти / Всі очікувані карти  
        - Показує, скільки очікуваних карт було знайдено
        - Приклад: якщо очікували 2 карти і знайшли 1, то Recall = 1/2 = 0.5
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        - Гармонічне середнє, яке балансує точність і повноту
        """
        retrieved_set = set(retrieved_cards)
        expected_set = set(expected_cards)
        
        # Кількість правильно витягнутих карт
        correct_cards = retrieved_set.intersection(expected_set)
        
        # Precision: скільки з витягнутих карт правильні
        precision = len(correct_cards) / len(retrieved_set) if retrieved_set else 0
        
        # Recall: скільки очікуваних карт знайдено
        recall = len(correct_cards) / len(expected_set) if expected_set else 0
        
        # F1 Score: гармонічне середнє
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "correct_cards": list(correct_cards),
            "missing_cards": list(expected_set - retrieved_set),
            "extra_cards": list(retrieved_set - expected_set)
        }
    
    def evaluate_keyword_presence(self, generated_text: str, expected_keywords: List[str]) -> Dict[str, float]:
        """Оцінка наявності ключових слів"""
        text_lower = generated_text.lower()
        
        found_keywords = []
        missing_keywords = []
        
        for keyword in expected_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        keyword_presence = len(found_keywords) / len(expected_keywords) if expected_keywords else 0
        
        return {
            "keyword_presence_score": keyword_presence,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "total_expected": len(expected_keywords),
            "total_found": len(found_keywords)
        }
    
    def evaluate_context(self, generated_text: str, expected_context: List[str]) -> Dict[str, float]:
        """Оцінка правильності контексту"""
        text_lower = generated_text.lower()
        
        found_context = []
        missing_context = []
        
        for context in expected_context:
            if context.lower() in text_lower:
                found_context.append(context)
            else:
                missing_context.append(context)
        
        context_score = len(found_context) / len(expected_context) if expected_context else 0
        
        return {
            "context_score": context_score,
            "found_context": found_context,
            "missing_context": missing_context
        }
    
    def evaluate_response_quality(self, response: str) -> Dict[str, float]:
        """Оцінка загальної якості відповіді"""
        words = response.split()
        unique_words = set(word.lower() for word in words)
        sentences = response.split('.')
        paragraphs = response.split('\n\n')
        
        return {
            "word_count": len(words),
            "unique_word_count": len(unique_words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "vocabulary_diversity": len(unique_words) / len(words) if words else 0
        }
    
    def save_results(self, results: Dict, output_file: str = "rag_evaluation_results.json"):
        """Збереження результатів оцінки"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Результати збережено в {output_file}")

async def run_evaluation():
    """Запуск процесу оцінки"""
    print("🔍 Початок оцінки RAG системи...")
    print("=" * 60)
    
    # Ініціалізація
    evaluator = RAGEvaluator()
    load_dotenv()
    agent = TarotAgent(
        cards_path="./app/static/images/cards",
        vector_store_path="./vector_store"
    )
    agent.initialize_vector_store()
    
    all_results = []
    total_metrics = {
        "f1_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "keyword_scores": [],
        "context_scores": []
    }
    
    # Проходимо по всіх тестових випадках
    for i, test_case in enumerate(evaluator.test_cases, 1):
        print(f"\n📋 Тест {i}/{len(evaluator.test_cases)}: {test_case['query']}")
        print("-" * 50)
        
        try:
            # Отримуємо відповідь від агента
            response = await agent.get_reading(test_case["query"])
            
            # Витягуємо карти та текст відповіді
            retrieved_cards = [card["name"] for card in response["cards"]]
            generated_text = response["reading"]
            
            # Оцінюємо різні аспекти
            retrieval_metrics = evaluator.evaluate_retrieval(retrieved_cards, test_case["expected_cards"])
            keyword_metrics = evaluator.evaluate_keyword_presence(generated_text, test_case["expected_keywords"])
            context_metrics = evaluator.evaluate_context(generated_text, test_case["expected_context"])
            quality_metrics = evaluator.evaluate_response_quality(generated_text)
            
            # Виводимо детальні результати
            print(f"🎴 Очікувані карти: {test_case['expected_cards']}")
            print(f"🎯 Витягнуті карти: {retrieved_cards}")
            
            print(f"\n📊 Метрики витягування:")
            print(f"   • Precision: {retrieval_metrics['precision']:.2f}")
            print(f"   • Recall: {retrieval_metrics['recall']:.2f}")
            print(f"   • F1 Score: {retrieval_metrics['f1_score']:.2f}")
            
            if retrieval_metrics['correct_cards']:
                print(f"   ✅ Правильні карти: {retrieval_metrics['correct_cards']}")
            if retrieval_metrics['missing_cards']:
                print(f"   ❌ Пропущені карти: {retrieval_metrics['missing_cards']}")
            if retrieval_metrics['extra_cards']:
                print(f"   ➕ Зайві карти: {retrieval_metrics['extra_cards']}")
            
            print(f"\n📝 Аналіз контенту:")
            print(f"   • Ключові слова: {keyword_metrics['keyword_presence_score']:.2f} ({keyword_metrics['total_found']}/{keyword_metrics['total_expected']})")
            print(f"   • Контекст: {context_metrics['context_score']:.2f}")
            
            if keyword_metrics['found_keywords']:
                print(f"   ✅ Знайдені слова: {keyword_metrics['found_keywords']}")
            if keyword_metrics['missing_keywords']:
                print(f"   ❌ Пропущені слова: {keyword_metrics['missing_keywords']}")
            
            print(f"\n📈 Якість відповіді:")
            print(f"   • Кількість слів: {quality_metrics['word_count']}")
            print(f"   • Різноманітність словника: {quality_metrics['vocabulary_diversity']:.2f}")
            print(f"   • Середня довжина речення: {quality_metrics['avg_words_per_sentence']:.1f} слів")
            
            # Зберігаємо метрики
            total_metrics["f1_scores"].append(retrieval_metrics['f1_score'])
            total_metrics["precision_scores"].append(retrieval_metrics['precision'])
            total_metrics["recall_scores"].append(retrieval_metrics['recall'])
            total_metrics["keyword_scores"].append(keyword_metrics['keyword_presence_score'])
            total_metrics["context_scores"].append(context_metrics['context_score'])
            
            # Зберігаємо детальні результати
            test_result = {
                "test_number": i,
                "query": test_case["query"],
                "expected_cards": test_case["expected_cards"],
                "retrieved_cards": retrieved_cards,
                "retrieval_metrics": retrieval_metrics,
                "keyword_metrics": keyword_metrics,
                "context_metrics": context_metrics,
                "quality_metrics": quality_metrics,
                "generated_text": generated_text
            }
            all_results.append(test_result)
            
        except Exception as e:
            print(f"❌ Помилка при обробці запиту: {str(e)}")
    
    # Обчислюємо середні значення
    print("\n" + "=" * 60)
    print("📊 ЗАГАЛЬНІ РЕЗУЛЬТАТИ")
    print("=" * 60)
    
    avg_f1 = np.mean(total_metrics["f1_scores"])
    avg_precision = np.mean(total_metrics["precision_scores"])
    avg_recall = np.mean(total_metrics["recall_scores"])
    avg_keywords = np.mean(total_metrics["keyword_scores"])
    avg_context = np.mean(total_metrics["context_scores"])
    
    print(f"🎯 Середні метрики витягування:")
    print(f"   • F1 Score: {avg_f1:.3f} ± {np.std(total_metrics['f1_scores']):.3f}")
    print(f"   • Precision: {avg_precision:.3f} ± {np.std(total_metrics['precision_scores']):.3f}")
    print(f"   • Recall: {avg_recall:.3f} ± {np.std(total_metrics['recall_scores']):.3f}")
    
    print(f"\n📝 Середні метрики контенту:")
    print(f"   • Ключові слова: {avg_keywords:.3f} ± {np.std(total_metrics['keyword_scores']):.3f}")
    print(f"   • Контекст: {avg_context:.3f} ± {np.std(total_metrics['context_scores']):.3f}")
    
    # Загальна оцінка
    overall_score = (avg_f1 + avg_keywords + avg_context) / 3
    print(f"\n🏆 Загальна оцінка RAG системи: {overall_score:.3f}")
    
    # Зберігаємо результати
    final_results = {
        "summary": {
            "average_f1_score": avg_f1,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_keyword_score": avg_keywords,
            "average_context_score": avg_context,
            "overall_score": overall_score,
            "total_tests": len(evaluator.test_cases)
        },
        "detailed_results": all_results
    }
    
    evaluator.save_results(final_results)
    print(f"\n💾 Детальні результати збережено в rag_evaluation_results.json")

def explain_f1_score():
    """Пояснення F1 Score"""
    print("\n" + "=" * 60)
    print("📚 ПОЯСНЕННЯ F1 SCORE")
    print("=" * 60)
    
    print("""
F1 Score - це метрика, яка поєднує точність (Precision) і повноту (Recall).

🎯 PRECISION (Точність):
   Precision = Правильно витягнуті карти / Всі витягнуті карти
   
   Приклад: Якщо система витягла 3 карти, а 2 з них правильні:
   Precision = 2/3 = 0.67 (67%)
   
   Високий Precision означає, що система рідко помиляється.

🔍 RECALL (Повнота):
   Recall = Правильно витягнуті карти / Всі очікувані карти
   
   Приклад: Якщо очікували 2 карти, а знайшли тільки 1:
   Recall = 1/2 = 0.5 (50%)
   
   Високий Recall означає, що система знаходить більшість релевантних результатів.

⚖️ F1 SCORE:
   F1 = 2 * (Precision * Recall) / (Precision + Recall)
   
   F1 Score балансує між точністю і повнотою:
   • F1 = 1.0 - ідеальний результат
   • F1 = 0.0 - найгірший результат
   • F1 > 0.8 - відмінний результат
   • F1 > 0.6 - добрий результат
   • F1 < 0.4 - потребує покращення

📊 ПРИКЛАДИ:
   
   Випадок 1: Висока точність, низька повнота
   • Витягли: ["The Magician"] 
   • Очікували: ["The Magician", "Death"]
   • Precision = 1/1 = 1.0, Recall = 1/2 = 0.5
   • F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.67
   
   Випадок 2: Низька точність, висока повнота  
   • Витягли: ["The Magician", "Death", "The Fool"]
   • Очікували: ["The Magician", "Death"] 
   • Precision = 2/3 = 0.67, Recall = 2/2 = 1.0
   • F1 = 2 * (0.67 * 1.0) / (0.67 + 1.0) = 0.8
   
   Випадок 3: Збалансований результат
   • Витягли: ["The Magician", "Death"]
   • Очікували: ["The Magician", "Death"]
   • Precision = 2/2 = 1.0, Recall = 2/2 = 1.0  
   • F1 = 2 * (1.0 * 1.0) / (1.0 + 1.0) = 1.0
""")

if __name__ == "__main__":
    import sys
    
    if "--explain" in sys.argv:
        explain_f1_score()
    else:
        asyncio.run(run_evaluation())
