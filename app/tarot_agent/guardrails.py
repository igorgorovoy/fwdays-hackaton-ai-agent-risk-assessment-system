"""
Guardrails система для Tarot AI агента
Забезпечує безпеку, якість та надійність відповідей
"""
import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
import json

@dataclass
class GuardrailResult:
    """Результат перевірки Guardrail"""
    is_safe: bool
    reason: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict = None

class TarotGuardrails:
    """Система Guardrails для Tarot AI агента"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Налаштування лімітів
        self.max_question_length = 500
        self.max_response_length = 5000
        self.min_confidence_threshold = 0.7
        
        # Заборонені слова/фрази для серйозних тем
        self.forbidden_words = [
            "самогубство", "suicide", "kill", "вбивство", "смерть людини",
            "наркотики", "drugs", "gambling", "азартні ігри",
            "терроризм", "terrorism", "насилля", "violence"
        ]
        
        # Дозволені теми для Таро
        self.allowed_topics = [
            "кар'єра", "стосунки", "фінанси", "здоров'я",
            "особистий розвиток", "духовність", "майбутнє",
            "career", "relationships", "money", "health"
        ]
        
        # Статистика
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "processing_times": [],
            "blocked_reasons": {}
        }

    def input_validation(self, question: str) -> GuardrailResult:
        """Валідація вхідного запиту"""
        self.stats["total_requests"] += 1
        
        # Перевірка на порожній запит
        if not question or not question.strip():
            return self._block_request("Порожній запит")
        
        # Перевірка довжини
        if len(question) > self.max_question_length:
            return self._block_request(f"Питання завелике (>{self.max_question_length} символів)")
        
        # Перевірка на заборонені слова
        question_lower = question.lower()
        for word in self.forbidden_words:
            if word in question_lower:
                return self._block_request(f"Виявлено заборонене слово: {word}")
        
        # Перевірка на спам (повторення символів)
        if self._is_spam(question):
            return self._block_request("Виявлено спам або некоректний текст")
        
        # Перевірка на SQL ін'єкції та XSS
        if self._has_malicious_content(question):
            return self._block_request("Виявлено підозрілий контент")
        
        return GuardrailResult(is_safe=True)

    def context_quality_check(self, retrieved_docs: List, question: str) -> GuardrailResult:
        """Перевірка якості витягнутого контексту"""
        
        if not retrieved_docs:
            return GuardrailResult(
                is_safe=False,
                reason="Не знайдено релевантних документів",
                confidence=1.0
            )
        
        # Перевірка релевантності (спрощена)
        relevance_score = self._calculate_relevance(retrieved_docs, question)
        
        if relevance_score < self.min_confidence_threshold:
            return GuardrailResult(
                is_safe=False,
                reason=f"Низька релевантність контексту: {relevance_score:.2f}",
                confidence=relevance_score
            )
        
        return GuardrailResult(
            is_safe=True,
            confidence=relevance_score,
            metadata={"relevance_score": relevance_score, "docs_count": len(retrieved_docs)}
        )

    def output_validation(self, response: str, cards: List[Dict]) -> GuardrailResult:
        """Валідація вихідної відповіді"""
        
        # Перевірка довжини відповіді
        if len(response) > self.max_response_length:
            return GuardrailResult(
                is_safe=False,
                reason=f"Відповідь завелика (>{self.max_response_length} символів)"
            )
        
        # Перевірка мінімальної довжини
        if len(response) < 50:
            return GuardrailResult(
                is_safe=False,
                reason="Відповідь занадто коротка"
            )
        
        # Перевірка на наявність заборонених слів у відповіді
        response_lower = response.lower()
        for word in self.forbidden_words:
            if word in response_lower:
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Відповідь містить заборонене слово: {word}"
                )
        
        # Перевірка структури відповіді
        if not self._has_valid_structure(response, cards):
            return GuardrailResult(
                is_safe=False,
                reason="Відповідь має некоректну структуру"
            )
        
        # Перевірка на галюцинації (спрощена)
        hallucination_score = self._detect_hallucinations(response, cards)
        if hallucination_score > 0.3:
            return GuardrailResult(
                is_safe=False,
                reason=f"Виявлено можливі галюцинації: {hallucination_score:.2f}",
                confidence=1.0 - hallucination_score
            )
        
        return GuardrailResult(is_safe=True, metadata={"response_length": len(response)})

    def rate_limiting(self, user_id: str = "default") -> GuardrailResult:
        """Обмеження частоти запитів"""
        # Спрощена реалізація - в реальному проекті використовувати Redis
        current_time = time.time()
        
        # Дозволяємо максимум 10 запитів на хвилину
        max_requests_per_minute = 10
        
        # Тут би була перевірка кількості запитів від користувача
        # За замовчуванням дозволяємо всі запити
        return GuardrailResult(is_safe=True, metadata={"user_id": user_id})

    def content_appropriateness(self, question: str) -> GuardrailResult:
        """Перевірка відповідності контенту Таро тематиці"""
        
        question_lower = question.lower()
        
        # Перевірка на медичні поради
        medical_keywords = ["хвороба", "лікування", "діагноз", "symptom", "cancer", "рак"]
        for keyword in medical_keywords:
            if keyword in question_lower:
                return GuardrailResult(
                    is_safe=False,
                    reason="Таро не може надавати медичні поради. Зверніться до лікаря."
                )
        
        # Перевірка на юридичні поради
        legal_keywords = ["суд", "закон", "legal", "lawsuit", "судова справа"]
        for keyword in legal_keywords:
            if keyword in question_lower:
                return GuardrailResult(
                    is_safe=False,
                    reason="Таро не може надавати юридичні поради. Зверніться до адвоката."
                )
        
        # Перевірка на фінансові поради
        financial_keywords = ["інвестиції", "біткоїн", "акції", "investment", "stock"]
        for keyword in financial_keywords:
            if keyword in question_lower:
                # Не блокуємо, але попереджаємо
                self.logger.warning(f"Фінансова тема в запиті: {question}")
        
        return GuardrailResult(is_safe=True)

    def _block_request(self, reason: str) -> GuardrailResult:
        """Допоміжний метод для блокування запиту"""
        self.stats["blocked_requests"] += 1
        self.stats["blocked_reasons"][reason] = self.stats["blocked_reasons"].get(reason, 0) + 1
        self.logger.warning(f"Запит заблоковано: {reason}")
        return GuardrailResult(is_safe=False, reason=reason)

    def _is_spam(self, text: str) -> bool:
        """Простий детектор спаму"""
        # Перевірка на багато повторюваних символів
        if re.search(r'(.)\1{5,}', text):
            return True
        
        # Перевірка на багато великих літер
        if len(text) > 10 and sum(1 for c in text if c.isupper()) / len(text) > 0.7:
            return True
        
        # Перевірка на багато чисел
        if len(text) > 20 and sum(1 for c in text if c.isdigit()) / len(text) > 0.5:
            return True
        
        return False

    def _has_malicious_content(self, text: str) -> bool:
        """Перевірка на шкідливий контент"""
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript injection
            r'SELECT.*FROM.*WHERE',  # SQL injection
            r'DROP\s+TABLE',  # SQL injection
            r'UNION\s+SELECT',  # SQL injection
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    def _calculate_relevance(self, docs: List, question: str) -> float:
        """Спрощений розрахунок релевантності"""
        if not docs:
            return 0.0
        
        question_words = set(question.lower().split())
        total_relevance = 0
        
        for doc in docs:
            doc_content = getattr(doc, 'page_content', str(doc))
            doc_words = set(doc_content.lower().split())
            intersection = question_words.intersection(doc_words)
            relevance = len(intersection) / len(question_words) if question_words else 0
            total_relevance += relevance
        
        return min(total_relevance / len(docs), 1.0)

    def _has_valid_structure(self, response: str, cards: List[Dict]) -> bool:
        """Перевірка структури відповіді"""
        # Перевірка наявності згадки принаймні половини карт
        mentioned_cards = 0
        for card in cards:
            card_name = card.get("name", "")
            if card_name and card_name in response:
                mentioned_cards += 1
        
        if mentioned_cards < len(cards) // 2:
            return False
        
        # Перевірка мінімальної довжини
        if len(response) < 100:
            return False
        
        # Перевірка наявності хоча б одного абзацу
        if '\n' not in response and len(response) > 200:
            return False
        
        return True

    def _detect_hallucinations(self, response: str, cards: List[Dict]) -> float:
        """Спрощений детектор галюцинацій"""
        hallucination_score = 0.0
        
        # Перевірка на згадку неіснуючих карт
        mentioned_cards = []
        for card in cards:
            card_name = card.get("name", "")
            if card_name and card_name in response:
                mentioned_cards.append(card_name)
        
        # Якщо згадано менше карт ніж очікувалось
        if len(mentioned_cards) < len(cards):
            hallucination_score += 0.1
        
        # Перевірка на згадку карт, яких немає в списку
        all_tarot_cards = [
            "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
            "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
            "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
            "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World"
        ]
        
        drawn_card_names = [card.get("name", "") for card in cards]
        
        for tarot_card in all_tarot_cards:
            if tarot_card in response and tarot_card not in drawn_card_names:
                hallucination_score += 0.15
        
        return min(hallucination_score, 1.0)

    def get_stats(self) -> Dict:
        """Отримання статистики Guardrails"""
        avg_processing_time = (
            sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
            if self.stats["processing_times"] else 0
        )
        
        return {
            "total_requests": self.stats["total_requests"],
            "blocked_requests": self.stats["blocked_requests"],
            "block_rate": (
                self.stats["blocked_requests"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            ),
            "average_processing_time": avg_processing_time,
            "blocked_reasons": self.stats["blocked_reasons"]
        }

    def reset_stats(self):
        """Скидання статистики"""
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "processing_times": [],
            "blocked_reasons": {}
        }

    def export_blocked_requests(self, filename: str = "blocked_requests.json"):
        """Експорт інформації про заблоковані запити"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.stats["blocked_reasons"], f, ensure_ascii=False, indent=2)

# Декоратор для застосування Guardrails
def apply_guardrails(guardrails_instance):
    """Декоратор для автоматичного застосування Guardrails"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Отримуємо питання з аргументів
            question = kwargs.get('question') or (args[1] if len(args) > 1 else '')
            
            try:
                # Застосовуємо input guardrails
                input_result = guardrails_instance.input_validation(question)
                if not input_result.is_safe:
                    raise ValueError(f"Input validation failed: {input_result.reason}")
                
                content_result = guardrails_instance.content_appropriateness(question)
                if not content_result.is_safe:
                    raise ValueError(f"Content validation failed: {content_result.reason}")
                
                rate_result = guardrails_instance.rate_limiting()
                if not rate_result.is_safe:
                    raise ValueError(f"Rate limiting triggered: {rate_result.reason}")
                
                # Виконуємо основну функцію
                result = await func(*args, **kwargs)
                
                # Застосовуємо output guardrails
                if isinstance(result, dict) and 'reading' in result:
                    output_result = guardrails_instance.output_validation(
                        result['reading'], 
                        result.get('cards', [])
                    )
                    if not output_result.is_safe:
                        # Логуємо, але не блокуємо - можна спробувати регенерувати
                        guardrails_instance.logger.warning(
                            f"Output validation warning: {output_result.reason}"
                        )
                
                # Записуємо час обробки
                processing_time = time.time() - start_time
                guardrails_instance.stats["processing_times"].append(processing_time)
                
                return result
                
            except Exception as e:
                # Логуємо помилку
                guardrails_instance.logger.error(f"Guardrails error: {str(e)}")
                raise
        
        return wrapper
    return decorator

# Приклад використання з контекстним менеджером
class GuardrailsContext:
    """Контекстний менеджер для Guardrails"""
    
    def __init__(self, guardrails: TarotGuardrails):
        self.guardrails = guardrails
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self.guardrails
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            processing_time = time.time() - self.start_time
            self.guardrails.stats["processing_times"].append(processing_time)
