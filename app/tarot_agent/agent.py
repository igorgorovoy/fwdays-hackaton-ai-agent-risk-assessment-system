"""
Main Tarot AI Agent module
"""
import os
import logging
from typing import List, Dict, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from .data_loader import TarotDataLoader
from .vector_store import TarotVectorStore
from .observability import TarotObservability

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TarotAgent:
    """Main Tarot AI Agent class"""
    
    def __init__(self, cards_path: str, vector_store_path: str):
        """Initialize Tarot Agent"""
        logger.info(f"Ініціалізація TarotAgent з параметрами:")
        logger.info(f"- cards_path: {cards_path}")
        logger.info(f"- vector_store_path: {vector_store_path}")
        
        self.cards_path = cards_path
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.retrieval_chain = None
        
        # Initialize observability
        self.observability = TarotObservability()
        
        # Initialize LLM
        logger.info("Ініціалізація LLM (GPT-4)...")
        try:
            from openai import OpenAI
            client = OpenAI()  # Це автоматично використає OPENAI_API_KEY з середовища
            
            self.llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                client=client
            )
            logger.info("LLM успішно ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка при ініціалізації LLM: {str(e)}")
            raise

    def _create_chains(self):
        """Create simple LLM chain for processing"""
        # Простий промпт без складних ланцюжків
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Ти таролог. Інтерпретуй карти базуючись на наданій інформації. 
            Відповідай українською, зрозуміло та професійно."""),
            ("human", "Питання: {input}\n\nКарти:\n{context}")
        ])
        
        # Простий ланцюжок: промпт + LLM
        self.simple_chain = self.prompt_template | self.llm
        
        # Залишаємо retrieval_chain для сумісності, але не використовуємо
        self.retrieval_chain = None

    def initialize_vector_store(self) -> None:
        """Initialize or update vector store with card data"""
        logger.info("Початок ініціалізації векторного сховища...")
        
        try:
            # Initialize vector store
            logger.info("Ініціалізація векторного сховища...")
            self.vector_store = TarotVectorStore(self.vector_store_path)
            
            # Load and prepare card data
            logger.info("Завантаження та підготовка даних карт...")
            loader = TarotDataLoader(self.cards_path)
            documents = loader.prepare_documents()
            logger.info(f"Підготовлено {len(documents)} документів для векторизації")
            
            # Create or update vector store
            logger.info("Створення/оновлення векторного сховища...")
            self.vector_store.create_or_update(documents)
            logger.info("Векторне сховище успішно оновлено")
            
            # Create the retrieval chain
            logger.info("Створення ланцюжка для обробки запитів...")
            self._create_chains()
            logger.info("Ланцюжок успішно створено")
            
        except Exception as e:
            logger.error(f"Помилка при ініціалізації векторного сховища: {str(e)}")
            raise

    def _get_card_path(self, card_name: str, is_reversed: bool = False) -> str:
        """Повертає шлях до зображення карти"""
        try:
            # Визначаємо директорію на основі типу карти
            if any(suit in card_name for suit in ['Cups', 'Pentacles', 'Swords', 'Wands']):
                suit = next(s for s in ['Cups', 'Pentacles', 'Swords', 'Wands'] if s in card_name)
                directory = f"MinorArcana_{suit}"
                
                # Для молодших арканів
                # Номери файлів відповідають порядку карт: Ace=0, Two=1, ..., Ten=9, Page=10, ...
                value = card_name.split(' ')[0]
                if value == 'Ace':
                    card_number = '0'
                elif value in ['Page', 'Knight', 'Queen', 'King']:
                    court_map = {'Page': '10', 'Knight': '11', 'Queen': '12', 'King': '13'}
                    card_number = court_map[value]
                else:
                    # Для числових карт (Two through Ten)
                    number_map = {
                        'Two': '1', 'Three': '2', 'Four': '3', 'Five': '4',
                        'Six': '5', 'Seven': '6', 'Eight': '7', 'Nine': '8', 'Ten': '9'
                    }
                    card_number = number_map.get(value, '0')
                
            else:
                directory = "MajorArcana"
                # Для старших арканів
                major_arcana_order = {
                    'The Fool': '0', 'The Magician': '1', 'The High Priestess': '2',
                    'The Empress': '3', 'The Emperor': '4', 'The Hierophant': '5',
                    'The Lovers': '6', 'The Chariot': '7', 'Strength': '8',
                    'The Hermit': '9', 'Wheel of Fortune': '10', 'Justice': '11',
                    'The Hanged Man': '12', 'Death': '13', 'Temperance': '14',
                    'The Devil': '15', 'The Tower': '16', 'The Star': '17',
                    'The Moon': '18', 'The Sun': '19', 'Judgement': '20',
                    'The World': '21'
                }
                card_number = major_arcana_order.get(card_name, '0')
            
            logger.info(f"Card mapping: {card_name} -> {directory}/{card_number}")
            
            # Формуємо базовий шлях
            base_path = f"/static/images/cards/{directory}/{card_number}"
            
            # Формуємо повні шляхи для перевірки існування файлів
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'cards', directory)
            regular_file = os.path.join(base_dir, f"{card_number}.jpg")
            reversed_file = os.path.join(base_dir, f"{card_number}-r.jpg")
            
            # Перевіряємо наявність файлів
            regular_exists = os.path.exists(regular_file)
            reversed_exists = os.path.exists(reversed_file)
            
            logger.info(f"Card: {card_name}, Number: {card_number}, Directory: {directory}")
            logger.info(f"Regular file exists: {regular_exists}, path: {regular_file}")
            logger.info(f"Reversed file exists: {reversed_exists}, path: {reversed_file}")
            
            # Формуємо відносні шляхи для веб
            regular_path = f"{base_path}.jpg"
            reversed_path = f"{base_path}-r.jpg"
            
            # Вибираємо правильний шлях
            if is_reversed and reversed_exists:
                image_path = reversed_path
                logger.info(f"Using reversed image: {image_path}")
            else:
                if is_reversed:
                    logger.warning(f"Reversed image not found: {reversed_path}, using regular image")
                image_path = regular_path
                logger.info(f"Using regular image: {image_path}")
            
            logger.info(f"Image path: {image_path}")
            return image_path
            
        except Exception as e:
            logger.error(f"Error in _get_card_path for {card_name}: {str(e)}")
            # Повертаємо шлях до карти The Fool як запасний варіант
            return "/static/images/cards/MajorArcana/0.jpg"

    def _draw_cards(self, num_cards: int = 3) -> List[Dict]:
        """Випадково вибирає карти для читання"""
        import random
        
        all_cards = [
            # Старші Аркани
            'The Fool', 'The Magician', 'The High Priestess', 'The Empress', 'The Emperor',
            'The Hierophant', 'The Lovers', 'The Chariot', 'Strength', 'The Hermit',
            'Wheel of Fortune', 'Justice', 'The Hanged Man', 'Death', 'Temperance',
            'The Devil', 'The Tower', 'The Star', 'The Moon', 'The Sun',
            'Judgement', 'The World',
            # Молодші Аркани - Кубки
            'Ace of Cups', 'Two of Cups', 'Three of Cups', 'Four of Cups', 'Five of Cups',
            'Six of Cups', 'Seven of Cups', 'Eight of Cups', 'Nine of Cups', 'Ten of Cups',
            'Page of Cups', 'Knight of Cups', 'Queen of Cups', 'King of Cups',
            # Молодші Аркани - Пентаклі
            'Ace of Pentacles', 'Two of Pentacles', 'Three of Pentacles', 'Four of Pentacles',
            'Five of Pentacles', 'Six of Pentacles', 'Seven of Pentacles', 'Eight of Pentacles',
            'Nine of Pentacles', 'Ten of Pentacles', 'Page of Pentacles', 'Knight of Pentacles',
            'Queen of Pentacles', 'King of Pentacles',
            # Молодші Аркани - Мечі
            'Ace of Swords', 'Two of Swords', 'Three of Swords', 'Four of Swords',
            'Five of Swords', 'Six of Swords', 'Seven of Swords', 'Eight of Swords',
            'Nine of Swords', 'Ten of Swords', 'Page of Swords', 'Knight of Swords',
            'Queen of Swords', 'King of Swords',
            # Молодші Аркани - Жезли
            'Ace of Wands', 'Two of Wands', 'Three of Wands', 'Four of Wands',
            'Five of Wands', 'Six of Wands', 'Seven of Wands', 'Eight of Wands',
            'Nine of Wands', 'Ten of Wands', 'Page of Wands', 'Knight of Wands',
            'Queen of Wands', 'King of Wands'
        ]
        
        # Перемішуємо карти
        random.shuffle(all_cards)
        
        # Вибираємо карти та їх положення
        drawn_cards = []
        for card_name in all_cards[:num_cards]:
            is_reversed = random.choice([True, False])
            drawn_cards.append({
                'name': card_name,
                'is_reversed': is_reversed,
                'image_path': self._get_card_path(card_name, is_reversed)
            })
        
        logger.info(f"Витягнуті карти: {drawn_cards}")
        return drawn_cards

    async def get_reading(self, question: str, num_cards: int = 3) -> Dict:
        """
        Get a tarot reading for a specific question
        
        Args:
            question: The user's question or topic for the reading
            cards: Optional list of specific cards to focus on
        
        Returns:
            str: The AI-generated tarot reading
        """
        try:
            logger.info(f"Starting reading for question: {question}")
            
            # Витягуємо карти
            drawn_cards = self._draw_cards(num_cards)
            logger.info(f"Drawn cards: {drawn_cards}")
            
            # Create trace
            session_start_time = self.observability.start_timer()
            trace_id = self.observability.create_trace(
                question=question,
                cards=drawn_cards,
                metadata={"num_cards": num_cards}
            )
            
            # Якщо не вдалося створити trace, продовжуємо без логування
            if trace_id is None:
                logger.warning("Failed to create trace, continuing without observability")
            
            # Збираємо контекст про карти
            cards_context = []
            final_cards = []
            
            for card in drawn_cards:
                logger.info(f"Getting info for card: {card}")
                card_info = self.get_card_info(card['name'])
                
                if card_info:
                    logger.info(f"Found info for card: {card['name']}")
                    # Вибираємо правильний контент в залежності від положення карти
                    if card['is_reversed']:
                        content = card_info.get('rmean', card_info['content'])
                        position = "перевернутому положенні"
                    else:
                        content = card_info.get('umean', card_info['content'])
                        position = "прямому положенні"
                    
                    cards_context.append(f"Карта {card['name']} в {position}:\n{content}")
                    final_cards.append(card)
                else:
                    logger.warning(f"No info found for card: {card['name']}")
                    # Якщо не знайдено інформацію про карту, витягуємо нову
                    while True:
                        new_card = self._draw_cards(1)[0]
                        if new_card['name'] != card['name'] and self.get_card_info(new_card['name']):
                            card_info = self.get_card_info(new_card['name'])
                            if new_card['is_reversed']:
                                content = card_info.get('rmean', card_info['content'])
                                position = "перевернутому положенні"
                            else:
                                content = card_info.get('umean', card_info['content'])
                                position = "прямому положенні"
                            
                            cards_context.append(f"Карта {new_card['name']} в {position}:\n{content}")
                            final_cards.append(new_card)
                            break
            
            # Формуємо запит з контекстом
            formatted_question = f"""
            Питання: {question}
            
            Розклад карт:
            {', '.join(card['name'] + (' (перевернута)' if card['is_reversed'] else '') for card in drawn_cards)}
            
            Інформація про карти:
            {'\n\n'.join(cards_context)}
            
            Надайте детальну інтерпретацію цього розкладу в контексті питання користувача.
            Поясніть значення кожної карти окремо, а потім як вони взаємодіють між собою.
            """
            
            logger.info("Formatted question prepared")
            logger.debug(f"Full formatted question: {formatted_question}")
            
            # Перевіряємо чи ініціалізовано retrieval chain
            if not self.simple_chain:
                logger.error("Simple chain is not initialized!")
                raise ValueError("Simple chain is not initialized")
            
            # Get the response from the chain
            logger.info("Sending request to LLM...")
            try:
                # Start timing for entire process
                total_start = self.observability.start_timer()
                
                # Start timing for retrieval
                retrieval_start = self.observability.start_timer()
                
                # Отримуємо документи через retriever
                retriever = self.vector_store.db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # Зменшуємо з 5 до 3 документів
                )
                retrieved_docs = await retriever.ainvoke(formatted_question)
                retrieval_time = self.observability.end_timer(retrieval_start)
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.3f}s")
                
                # Start timing for LLM call
                llm_start = self.observability.start_timer()
                
                # Створюємо оптимізований контекст з отриманих документів
                context_parts = []
                for doc in retrieved_docs:
                    # Обрізаємо кожен документ до 150 символів для економії токенів
                    short_content = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                    context_parts.append(short_content)
                
                context = "\n\n".join(context_parts)
                
                logger.info(f"📄 Context length: {len(context)} chars (~{len(context.split())} words)")
                
                # Викликаємо простий ланцюжок
                response_obj = await self.simple_chain.ainvoke({
                    "input": formatted_question,
                    "context": context
                })
                
                response_text = response_obj.content
                llm_execution_time = self.observability.end_timer(llm_start)
                total_time = self.observability.end_timer(total_start)
                
                logger.info(f"LLM responded in {llm_execution_time:.3f}s (total: {total_time:.3f}s)")
                
                # Оцінюємо використання токенів (приблизно) 
                question_tokens = len(formatted_question.split()) * 1.3
                context_tokens = len(context.split()) * 1.3
                system_prompt_tokens = 20  # Приблизно для скороченого промпта
                
                prompt_tokens = question_tokens + context_tokens + system_prompt_tokens
                completion_tokens = len(response_text.split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
                
                logger.info(f"🔢 Token breakdown:")
                logger.info(f"   📝 Question: ~{int(question_tokens)} tokens")
                logger.info(f"   📄 Context: ~{int(context_tokens)} tokens") 
                logger.info(f"   ⚙️ System: ~{int(system_prompt_tokens)} tokens")
                logger.info(f"   📤 Prompt total: ~{int(prompt_tokens)} tokens")
                logger.info(f"   📥 Completion: ~{int(completion_tokens)} tokens")
                
                token_usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(total_tokens)
                }
                
                # Обчислюємо вартість
                estimated_cost = self.observability.calculate_cost(token_usage, "gpt-4-turbo-preview")
                
                logger.info(f"💰 Estimated cost: ${estimated_cost:.4f} (tokens: {int(total_tokens)})")
                logger.info(f"⏱️ Timing: retrieval={retrieval_time:.3f}s, llm={llm_execution_time:.3f}s, total={total_time:.3f}s")
                
                # Створюємо детальні метрики
                detailed_metrics = {
                    # ⏱️ Метрики продуктивності  
                    "retrieval_time_seconds": round(retrieval_time, 3),
                    "llm_execution_time_seconds": round(llm_execution_time, 3), 
                    "total_execution_time_seconds": round(total_time, 3),
                    "retrieval_time_ms": round(retrieval_time * 1000),
                    "llm_execution_time_ms": round(llm_execution_time * 1000),
                    "total_execution_time_ms": round(total_time * 1000),
                    
                    # 💰 Економічні метрики - це найважливіше!
                    "COST_USD": round(estimated_cost, 6),  # Робимо назву помітнішою
                    "estimated_cost_usd": round(estimated_cost, 6),
                    "cost_in_cents": round(estimated_cost * 100, 2),
                    "prompt_tokens": int(token_usage["prompt_tokens"]),
                    "completion_tokens": int(token_usage["completion_tokens"]), 
                    "total_tokens": int(token_usage["total_tokens"]),
                    "cost_per_1k_tokens": round(estimated_cost / (total_tokens / 1000), 6) if total_tokens > 0 else 0,
                    
                    # 🎴 Метрики контексту
                    "documents_retrieved": len(retrieved_docs),
                    "model_used": "gpt-4-turbo-preview",
                    "num_cards_drawn": num_cards,
                    "cards_drawn": [f"{card['name']} ({'R' if card['is_reversed'] else 'U'})" for card in drawn_cards],
                    
                    # 📊 Метрики якості
                    "context_length_chars": len(context),
                    "response_length_chars": len(response_text),
                    "processing_success": True,
                    "session_id": trace_id
                }
                
                # Log основних метрик сесії безпосередньо до trace
                if trace_id:
                    try:
                        # Спочатку оновлюємо outputs
                        self.observability.client.update_run(
                            run_id=trace_id,
                            outputs={
                                "response": response_text,
                                "metrics_summary": f"Cost: ${estimated_cost:.4f}, Time: {total_time:.1f}s, Tokens: {int(total_tokens)}",
                                "cards_info": [f"{card['name']} ({'перевернута' if card['is_reversed'] else 'пряма'})" 
                                              for card in drawn_cards]
                            }
                        )
                        
                        # Потім додаємо всі метрики до extra
                        self.observability.client.update_run(
                            run_id=trace_id,
                            extra=detailed_metrics
                        )
                        
                        logger.info("✅ Metrics successfully logged to trace")
                        logger.info(f"📊 Key metrics: Cost=${estimated_cost:.4f}, Time={total_time:.1f}s, Tokens={int(total_tokens)}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to update trace with metrics: {e}")
                        # Якщо не вдалося оновити через API, виводимо метрики в логи
                        logger.info("📊 МЕТРИКИ (fallback):")
                        for key, value in detailed_metrics.items():
                            logger.info(f"   {key}: {value}")
                
                response = {"answer": response_text}
            except Exception as chain_error:
                logger.error(f"Error during chain invocation: {str(chain_error)}")
                # Log error
                self.observability.log_error(
                    trace_id=trace_id,
                    error=chain_error,
                    context={"question": question, "cards": drawn_cards}
                )
                raise
            
            if not response:
                logger.error("Empty response received from LLM")
                raise ValueError("Empty response from LLM")
            
            if "answer" not in response:
                logger.error(f"No 'answer' in response. Keys present: {response.keys()}")
                raise ValueError("No 'answer' in LLM response")
            
            logger.info("Successfully generated reading")
            
            # Фінальне логування метрик сесії
            if trace_id:
                try:
                    session_duration = self.observability.end_timer(session_start_time)
                    self.observability.client.update_run(
                        run_id=trace_id,
                        extra={
                            **self.observability.client.read_run(trace_id).extra,
                            "session_duration_seconds": session_duration,
                            "session_success": True
                        }
                    )
                    logger.info(f"Session completed successfully in {session_duration:.3f}s")
                except Exception as finalize_error:
                    logger.warning(f"Failed to finalize trace: {str(finalize_error)}")
            
            return {
                "cards": final_cards,
                "reading": response["answer"]
            }
            
        except Exception as e:
            logger.error(f"Error in get_reading: {str(e)}", exc_info=True)
            
            # Логування невдалої сесії
            if 'trace_id' in locals() and trace_id:
                try:
                    session_duration = self.observability.end_timer(session_start_time)
                    self.observability.client.update_run(
                        run_id=trace_id,
                        extra={
                            "session_duration_seconds": session_duration,
                            "session_success": False,
                            "error_message": str(e)
                        }
                    )
                except Exception as finalize_error:
                    logger.warning(f"Failed to finalize error trace: {str(finalize_error)}")
            
            raise

    def get_card_info(self, card_name: str) -> Dict:
        """
        Get detailed information about a specific card
        
        Args:
            card_name: Name of the card to look up
            
        Returns:
            Dict: Detailed information about the card
        """
        # Search for card information
        docs = self.vector_store.similarity_search(
            f"Детальна інформація про карту {card_name}",
            k=1
        )
        
        if docs:
            return {
                'content': docs[0].page_content,
                'metadata': docs[0].metadata
            }
        return None
