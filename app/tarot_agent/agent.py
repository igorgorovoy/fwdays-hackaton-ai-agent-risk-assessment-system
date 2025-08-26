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
        """Create LangChain chains for processing"""
        # Create the prompt for processing retrieved documents
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ти - досвідчений таролог, який допомагає людям зрозуміти значення карт Таро.
            Твоя роль - надавати глибокі та змістовні інтерпретації карт, базуючись на їх традиційних значеннях.
            
            Важливі правила:
            1. Використовуй надану інформацію про карти
            2. Говори впевнено та професійно
            3. Пояснюй значення зрозуміло та доступно
            4. Зв'язуй значення карт з контекстом питання
            5. Відповідай українською мовою
            """),
            ("human", "Питання користувача: {input}\n\nКонтекст:\n{context}"),
            ("assistant", """Я проаналізую це питання та надану інформацію про карти.

Ось моя інтерпретація:""")
        ])

        # Create the document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_variable_name="context",
        )

        # Create the retrieval chain
        self.retrieval_chain = create_retrieval_chain(
            retriever=self.vector_store.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # повертаємо 5 найбільш релевантних документів
            ),
            combine_docs_chain=document_chain
        )

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
                value_map = {
                    'Ace': '0', 'Two': '1', 'Three': '2', 'Four': '3', 'Five': '4',
                    'Six': '5', 'Seven': '6', 'Eight': '7', 'Nine': '8', 'Ten': '9',
                    'Page': '10', 'Knight': '11', 'Queen': '12', 'King': '13'
                }
                value = card_name.split(' ')[0]
                card_number = value_map.get(value, '0')
                
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
            
            # Формуємо шлях до файлу
            base_path = f"/static/images/cards/{directory}/{card_number}"
            
            # Перевіряємо наявність файлу для перевернутої карти
            import os
            full_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'cards', directory, f"{card_number}-r.jpg")
            
            # Формуємо шляхи до файлів
            regular_path = f"{base_path}.jpg"
            reversed_path = f"{base_path}-r.jpg"
            
            # Перевіряємо наявність файлів
            regular_exists = os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'cards', directory, f"{card_number}.jpg"))
            reversed_exists = os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'static', 'images', 'cards', directory, f"{card_number}-r.jpg"))
            
            logger.info(f"Regular file exists: {regular_exists}, path: {regular_path}")
            logger.info(f"Reversed file exists: {reversed_exists}, path: {reversed_path}")
            
            # Вибираємо правильний шлях
            if is_reversed:
                if reversed_exists:
                    image_path = reversed_path
                else:
                    logger.warning(f"Reversed image not found: {reversed_path}, using regular image")
                    image_path = regular_path
            else:
                image_path = regular_path
            
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
            if not self.retrieval_chain:
                logger.error("Retrieval chain is not initialized!")
                raise ValueError("Retrieval chain is not initialized")
            
            # Get the response from the chain
            logger.info("Sending request to LLM...")
            try:
                response = await self.retrieval_chain.ainvoke({
                    "input": formatted_question  # змінено з "question" на "input"
                })
                logger.info("Received response from LLM")
                logger.debug(f"Raw response: {response}")
            except Exception as chain_error:
                logger.error(f"Error during chain invocation: {str(chain_error)}")
                raise
            
            if not response:
                logger.error("Empty response received from LLM")
                raise ValueError("Empty response from LLM")
            
            if "answer" not in response:
                logger.error(f"No 'answer' in response. Keys present: {response.keys()}")
                raise ValueError("No 'answer' in LLM response")
            
            logger.info("Successfully generated reading")
            return {
                "cards": final_cards,
                "reading": response["answer"]
            }
            
        except Exception as e:
            logger.error(f"Error in get_reading: {str(e)}", exc_info=True)
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
