# Tarot AI Agent

Tarot AI Agent - це Flask-додаток, який використовує LangChain та RAG (Retrieval Augmented Generation) для генерації інтерпретацій карт Таро на основі структурованих описів.

## Архітектура

### Діаграма класів

```mermaid
classDiagram
    class FlaskApp {
        +routes: Routes
        +templates: Templates
        +static: Static
    }
    class TarotAgent {
        +cards_path: str
        +vector_store_path: str
        +llm: ChatOpenAI
        +vector_store: TarotVectorStore
        +retrieval_chain: Chain
        +initialize_vector_store()
        +get_reading(question: str)
        +get_card_info(card_name: str)
        +_draw_cards(num_cards: int)
    }
    class TarotVectorStore {
        +db: Chroma
        +embeddings: SentenceTransformers
        +create_or_update(documents: List)
        +similarity_search(query: str)
    }
    class TarotDataLoader {
        +cards_path: str
        +prepare_documents()
        +_read_card_files()
    }
    class ChromaDB {
        +collection: Collection
        +persist()
    }
    class OpenAI {
        +model: str
        +temperature: float
    }

    FlaskApp --> TarotAgent: uses
    TarotAgent --> TarotVectorStore: manages
    TarotAgent --> OpenAI: uses
    TarotVectorStore --> ChromaDB: uses
    TarotAgent --> TarotDataLoader: uses
```

### Діаграма послідовності

```mermaid
sequenceDiagram
    participant User
    participant Flask as Flask App
    participant Agent as Tarot Agent
    participant RAG as RAG System
    participant LLM as GPT-4
    participant DB as Vector DB

    User->>Flask: Задає питання
    Flask->>Agent: Передає питання
    Agent->>Agent: Витягує випадкові карти
    Agent->>DB: Шукає інформацію про карти
    DB-->>Agent: Повертає описи карт
    Agent->>RAG: Формує контекст з описів
    RAG->>LLM: Запит з контекстом
    LLM-->>RAG: Генерує відповідь
    RAG-->>Agent: Повертає інтерпретацію
    Agent-->>Flask: Повертає карти та читання
    Flask-->>User: Показує результат
```

### Діаграма процесів

```mermaid
flowchart TD
    A[Початок] --> B[Ініціалізація Flask]
    B --> C[Завантаження конфігурації]
    C --> D[Ініціалізація TarotAgent]
    
    D --> E[Завантаження LLM]
    D --> F[Ініціалізація Vector Store]
    
    F --> G[Завантаження документів]
    G --> H[Створення ембедінгів]
    H --> I[Збереження в ChromaDB]
    
    E & I --> J[Створення ланцюжків]
    J --> K[Готовий до роботи]
    
    K --> L{Отримання запиту}
    L --> M[Вибір карт]
    M --> N[Пошук описів]
    N --> O[Формування контексту]
    O --> P[Генерація відповіді]
    P --> Q[Форматування результату]
    Q --> R[Відправка відповіді]
    R --> L
```

## High Level Design (HLD)

### Огляд системної архітектури

Система оцінки ризиків з AI агентом на основі Таро карт представляє собою багаторівневу архітектуру, що забезпечує надійну та масштабовану обробку запитів користувачів.

```mermaid
graph TB
    %% User Interface Layer
    subgraph "Frontend Layer"
        UI[Web Interface]
        API[REST API Endpoints]
    end

    %% Application Layer
    subgraph "Application Layer"
        Flask[Flask Application]
        Routes[Routes Handler]
        Agent[Tarot AI Agent]
    end

    %% AI & ML Services
    subgraph "AI/ML Services"
        LLM[OpenAI GPT-4<br/>Turbo Preview]
        RAG[RAG System<br/>LangChain]
        Embeddings[SentenceTransformers<br/>all-MiniLM-L6-v2]
    end

    %% Data Layer
    subgraph "Data Layer"
        VectorStore[TarotVectorStore<br/>ChromaDB]
        CardData[Card Data<br/>Text Files]
        Images[Card Images<br/>Static Assets]
    end

    %% Monitoring & Observability
    subgraph "Observability"
        Monitor[TarotObservability]
        Logs[Logging System]
    end

    %% External Dependencies
    subgraph "External Services"
        OpenAI[OpenAI API]
    end

    %% User Flow
    User((User)) --> UI
    UI --> API
    API --> Routes
    Routes --> Agent

    %% Agent Processing
    Agent --> RAG
    Agent --> VectorStore
    RAG --> LLM
    LLM --> OpenAI

    %% Data Flow
    CardData --> VectorStore
    VectorStore --> Embeddings
    Images --> UI

    %% Monitoring
    Agent --> Monitor
    Monitor --> Logs

    %% Return Path
    OpenAI --> LLM
    LLM --> RAG
    RAG --> Agent
    Agent --> Routes
    Routes --> API
    API --> UI
    UI --> User

    %% Styling
    classDef frontend fill:#e1f5fe
    classDef application fill:#f3e5f5
    classDef ai fill:#fff3e0
    classDef data fill:#e8f5e8
    classDef monitoring fill:#fce4ec
    classDef external fill:#f5f5f5

    class UI,API frontend
    class Flask,Routes,Agent application
    class LLM,RAG,Embeddings ai
    class VectorStore,CardData,Images data
    class Monitor,Logs monitoring
    class OpenAI external
```

### Детальна системна архітектура

```mermaid
graph TB
    subgraph "Client Tier"
        Browser[Web Browser]
        Mobile[Mobile Device]
    end

    subgraph "Presentation Tier"
        WebServer[Flask Web Server<br/>Port 8080]
        StaticAssets[Static Assets<br/>- CSS/JS<br/>- Card Images<br/>- Fonts]
        Templates[Jinja2 Templates<br/>index.html]
    end

    subgraph "Business Logic Tier"
        subgraph "Core Services"
            TarotAgent[Tarot Agent<br/>Main Controller]
            DataLoader[Tarot Data Loader<br/>Text Processing]
            Observability[Observability Service<br/>Monitoring & Logs]
        end
        
        subgraph "AI Processing Pipeline"
            RAGChain[RAG Chain<br/>Document Retrieval]
            LLMChain[LLM Chain<br/>Response Generation]
            CardSelector[Card Selection Logic<br/>Random Draw Algorithm]
        end
    end

    subgraph "AI/ML Services"
        subgraph "Language Models"
            GPT4[OpenAI GPT-4 Turbo<br/>Temperature: 0.7]
            Embeddings[SentenceTransformers<br/>all-MiniLM-L6-v2]
        end
        
        subgraph "Prompt Engineering"
            SystemPrompt[System Prompt<br/>Tarot Expert Persona]
            ContextTemplate[Context Template<br/>Ukrainian Language]
        end
    end

    subgraph "Data Persistence Tier"
        subgraph "Vector Database"
            ChromaDB[(ChromaDB<br/>Vector Store)]
            Metadata[Metadata Store<br/>Card Properties]
        end
        
        subgraph "File System"
            CardDescriptions[Card Descriptions<br/>468 .txt files]
            CardImages[Card Images<br/>206 .jpg files]
            VectorIndex[Vector Index<br/>Persistent Storage]
        end
    end

    subgraph "Configuration & Environment"
        EnvConfig[Environment Config<br/>.env file]
        Requirements[Dependencies<br/>requirements.txt]
        VenvPython[Python Virtual Env<br/>Python 3.12]
    end

    %% User Interactions
    Browser --> WebServer
    Mobile --> WebServer
    
    %% Web Server Routing
    WebServer --> Templates
    WebServer --> StaticAssets
    WebServer --> TarotAgent
    
    %% Core Service Interactions
    TarotAgent --> DataLoader
    TarotAgent --> Observability
    TarotAgent --> RAGChain
    TarotAgent --> CardSelector
    
    %% AI Pipeline
    RAGChain --> LLMChain
    RAGChain --> ChromaDB
    LLMChain --> GPT4
    LLMChain --> SystemPrompt
    GPT4 --> ContextTemplate
    
    %% Data Flow
    DataLoader --> CardDescriptions
    DataLoader --> Embeddings
    Embeddings --> ChromaDB
    ChromaDB --> VectorIndex
    ChromaDB --> Metadata
    
    %% Static Assets
    CardSelector --> CardImages
    StaticAssets --> CardImages
    
    %% Configuration
    TarotAgent --> EnvConfig
    WebServer --> VenvPython
    VenvPython --> Requirements
    
    %% Styling
    classDef client fill:#e3f2fd
    classDef presentation fill:#f3e5f5
    classDef business fill:#fff3e0
    classDef ai fill:#e8f5e8
    classDef data fill:#fce4ec
    classDef config fill:#f5f5f5

    class Browser,Mobile client
    class WebServer,StaticAssets,Templates presentation
    class TarotAgent,DataLoader,Observability,RAGChain,LLMChain,CardSelector business
    class GPT4,Embeddings,SystemPrompt,ContextTemplate ai
    class ChromaDB,Metadata,CardDescriptions,CardImages,VectorIndex data
    class EnvConfig,Requirements,VenvPython config
```

### Потік обробки запиту

```mermaid
sequenceDiagram
    participant User as Користувач
    participant Browser as Веб-браузер
    participant Flask as Flask App
    participant Agent as Tarot Agent
    participant Selector as Card Selector
    participant VectorDB as ChromaDB
    participant RAG as RAG Chain
    participant LLM as GPT-4 API
    participant Monitor as Observability

    Note over User,Monitor: Процес отримання Таро читання

    User->>Browser: Вводить питання
    Browser->>Flask: POST /api/reading<br/>{"question": "питання"}
    
    Flask->>Agent: get_reading(question)
    Agent->>Monitor: Логування початку запиту
    
    Note over Agent,Selector: Вибір карт
    Agent->>Selector: _draw_cards(num_cards=3)
    Selector-->>Agent: Список карт + положення
    
    Note over Agent,VectorDB: Пошук інформації про карти
    loop Для кожної карти
        Agent->>VectorDB: similarity_search(card_name)
        VectorDB-->>Agent: Описи карти (top-k=5)
    end
    
    Note over Agent,RAG: Формування контексту
    Agent->>RAG: Створення контексту з описів
    RAG->>Agent: Підготовлений prompt
    
    Note over Agent,LLM: Генерація інтерпретації
    Agent->>LLM: Запит з контекстом<br/>(температура: 0.7)
    LLM-->>Agent: Інтерпретація українською
    
    Note over Agent,Monitor: Обробка результату
    Agent->>Monitor: Логування успішної відповіді
    Agent->>Agent: Формування відповіді<br/>+ шляхи до зображень
    
    Agent-->>Flask: {"cards": [...], "reading": "..."}
    Flask-->>Browser: JSON відповідь
    Browser-->>User: Відображення карт<br/>та інтерпретації
    
    Note over User,Monitor: Завершення запиту
    
    alt Помилка в процесі
        Agent->>Monitor: Логування помилки
        Agent-->>Flask: Повернення помилки
        Flask-->>Browser: Error 500
        Browser-->>User: Повідомлення про помилку
    end
```

### Компонентна архітектура

```mermaid
graph TD
    subgraph "Tarot AI Risk Assessment System"
        subgraph "Web Application"
            FlaskApp[Flask Application<br/>- routes.py<br/>- templates/<br/>- static/]
            WebUI[Web Interface<br/>- Adaptive Design<br/>- Card Display<br/>- Ukrainian UI]
        end

        subgraph "AI Agent Core"
            TarotAgent[Tarot Agent<br/>- Card Selection<br/>- Reading Generation<br/>- Error Handling]
            
            DataLoader[Data Loader<br/>- Text File Processing<br/>- Document Preparation<br/>- Metadata Extraction]
            
            VectorStore[Vector Store<br/>- ChromaDB Interface<br/>- Similarity Search<br/>- Document Storage]
        end

        subgraph "AI/ML Pipeline"
            RAGSystem[RAG System<br/>- Retrieval Chain<br/>- Context Formation<br/>- Document Combination]
            
            LLMInterface[LLM Interface<br/>- GPT-4 Integration<br/>- Prompt Templates<br/>- Response Processing]
            
            Embeddings[Embedding Service<br/>- SentenceTransformers<br/>- Vector Generation<br/>- Text Encoding]
        end

        subgraph "Data Storage"
            CardDatabase[Card Database<br/>- 78 Tarot Cards<br/>- Multiple Descriptions<br/>- Upright/Reversed]
            
            VectorDB[Vector Database<br/>- ChromaDB<br/>- Persistent Storage<br/>- Similarity Index]
            
            ImageAssets[Image Assets<br/>- Card Images<br/>- Multiple Decks<br/>- Static Files]
        end

        subgraph "Infrastructure"
            Observability[Observability<br/>- Logging<br/>- Monitoring<br/>- Error Tracking]
            
            Configuration[Configuration<br/>- Environment Variables<br/>- API Keys<br/>- Settings]
            
            Dependencies[Dependencies<br/>- Python Packages<br/>- Virtual Environment<br/>- Requirements]
        end
    end

    subgraph "External Services"
        OpenAIAPI[OpenAI API<br/>- GPT-4 Turbo<br/>- Rate Limiting<br/>- Authentication]
    end

    subgraph "Development Tools"
        InitDB[Database Initialization<br/>- init_db.py<br/>- verify_db.py<br/>- check_documents.py]
        
        Testing[Testing Suite<br/>- test_rag.py<br/>- RAG Testing<br/>- Integration Tests]
    end

    %% Main connections
    FlaskApp --> TarotAgent
    TarotAgent --> DataLoader
    TarotAgent --> VectorStore
    TarotAgent --> RAGSystem
    TarotAgent --> Observability

    DataLoader --> CardDatabase
    DataLoader --> Embeddings
    VectorStore --> VectorDB
    RAGSystem --> LLMInterface
    RAGSystem --> VectorStore

    LLMInterface --> OpenAIAPI
    Embeddings --> VectorDB
    
    WebUI --> ImageAssets
    TarotAgent --> Configuration
    
    %% Development connections
    InitDB --> VectorDB
    InitDB --> CardDatabase
    Testing --> TarotAgent
    
    %% Infrastructure connections
    FlaskApp --> Dependencies
    TarotAgent --> Dependencies

    %% Styling
    classDef webapp fill:#e1f5fe,stroke:#01579b
    classDef aicore fill:#f3e5f5,stroke:#4a148c
    classDef mlpipeline fill:#fff3e0,stroke:#e65100
    classDef datastorage fill:#e8f5e8,stroke:#1b5e20
    classDef infrastructure fill:#fce4ec,stroke:#880e4f
    classDef external fill:#f5f5f5,stroke:#424242
    classDef devtools fill:#fff8e1,stroke:#f57f17

    class FlaskApp,WebUI webapp
    class TarotAgent,DataLoader,VectorStore aicore
    class RAGSystem,LLMInterface,Embeddings mlpipeline
    class CardDatabase,VectorDB,ImageAssets datastorage
    class Observability,Configuration,Dependencies infrastructure
    class OpenAIAPI external
    class InitDB,Testing devtools
```

### Архітектурні принципи

1. **Модульність**: Система розділена на чіткі компоненти з визначеними обов'язками
2. **Масштабованість**: Використання векторних баз даних та асинхронної обробки
3. **Спостережуваність**: Вбудована система моніторингу та логування
4. **Безпека**: Керування API ключами через змінні середовища
5. **Підтримуваність**: Чітка структура коду та документація

### Технічні характеристики HLD

#### Продуктивність
- **Час відповіді**: 3-7 секунд на запит
- **Конкурентність**: Асинхронна обробка запитів
- **Пропускна здатність**: До 10 одночасних користувачів
- **Кешування**: Векторні ембедінги зберігаються локально

#### Надійність
- **Error Handling**: Обробка помилок на всіх рівнях
- **Logging**: Детальне логування для діагностики
- **Fallback**: Резервні механізми при збоях API
- **Validation**: Валідація вхідних даних

#### Безпека
- **API Keys**: Зберігання в змінних середовища
- **Input Validation**: Перевірка користувацьких даних
- **Error Disclosure**: Запобігання витоку системної інформації
- **Rate Limiting**: Контроль частоти запитів

### Компоненти системи

1. **Flask Backend**
   - Обробка HTTP-запитів
   - Маршрутизація
   - Інтеграція з AI-агентом

2. **AI Agent (LangChain)**
   - Векторна база даних (ChromaDB)
   - Ембедінги (SentenceTransformers)
   - LLM (GPT-4-turbo-preview)
   - Ланцюжки обробки запитів

3. **Frontend**
   - Адаптивний веб-інтерфейс
   - Автоматичний вибір карт
   - Відображення карт та інтерпретацій

### Структура проекту

```
tarot/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── tarot_agent/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── data_loader.py
│   │   └── vector_store.py
│   ├── static/
│   │   ├── css/
│   │   ├── images/cards/
│   │   └── js/
│   └── templates/
│       └── index.html
├── vector_store/          # Ігнорується в git
├── requirements.txt
├── run.py
├── init_db.py
└── verify_db.py
```

## Встановлення та налаштування

1. **Клонування репозиторію**
   ```bash
   git clone https://github.com/your-username/tarot.git
   cd tarot
   ```

2. **Створення віртуального середовища**
   ```bash
   python -m venv venv
   source venv/bin/activate  # для Unix
   # або
   venv\Scripts\activate  # для Windows
   ```

3. **Встановлення залежностей**
   ```bash
   pip install -r requirements.txt
   ```

4. **Налаштування середовища**
   - Скопіюйте `.env.example` в `.env`
   - Додайте свій OpenAI API ключ
   ```bash
   cp env.example .env
   # Відредагуйте .env файл, додавши свій OPENAI_API_KEY
   ```

## Ініціалізація векторної бази

1. **Запуск ініціалізації**
   ```bash
   python init_db.py
   ```
   Скрипт:
   - Завантажить описи карт з текстових файлів
   - Створить ембедінги
   - Збереже їх у ChromaDB

2. **Перевірка ініціалізації**
   ```bash
   python verify_db.py
   ```
   Скрипт перевірить:
   - Кількість завантажених документів
   - Наявність описів для всіх карт
   - Коректність метаданих

## Запуск додатку

1. **Локальний запуск**
   ```bash
   python run.py
   ```
   Додаток буде доступний за адресою: http://localhost:8080

2. **Docker (опціонально)**
   ```bash
   docker-compose up --build
   ```

## Тестування

### Тестування RAG (test_rag.py)

Скрипт для тестування роботи RAG системи та генерації відповідей:
```bash
python test_rag.py
```

**Що тестується:**
1. Ініціалізація агента:
   - Завантаження моделі LLM
   - Підключення до векторної бази
   - Створення ланцюжків обробки

2. Обробка запитів:
   - "Що означає карта Маг?"
   - "Розкажи про карту Смерть"
   - "Що символізує Колесо Фортуни?"
   - "Опиши значення карти Місяць"
   - "Що означає Туз Кубків?"

3. Функціональність:
   - Автоматичний вибір карт
   - Визначення положення карт (пряме/перевернуте)
   - Генерація шляхів до зображень
   - Форматування відповідей

**Приклад виводу:**
```
Запит: Що означає карта Маг?

Витягнуті карти:
- The World (перевернута)
- The Hermit (пряма)
- The Devil (перевернута)

Відповідь:
[Детальна інтерпретація карт...]
```

### Аналіз документів (check_documents.py)

Скрипт для глибокого аналізу завантажених документів у векторній базі:
```bash
python check_documents.py
```

**Функціональність:**
1. Перевірка вибраних карт:
   - Старші Аркани:
     - The Fool
     - The Magician
     - Death
     - The World
   - Молодші Аркани:
     - Ace of Cups
     - King of Pentacles
     - Queen of Swords
     - Knight of Wands

2. Аналіз метаданих:
   - type: тип карти (major/minor)
   - name: назва карти
   - suit: масть (для молодших арканів)
   - aspect: положення карти (upright/reversed)

3. Статистика:
   - Загальна кількість документів
   - Розподіл за типами:
     - Старші Аркани: ~118 документів
     - Молодші Аркани: ~882 документи

**Приклад виводу:**
```
=== The Fool ===
Знайдено інформацію:
Контент: The Fool is always whole, healthy and without fear...
Метадані: {'aspect': 'upright', 'name': 'Fool', 'suit': 'NA', 'type': 'major'}

Загальна кількість документів: 1000
Розподіл документів за типами:
- minor: 882 документів
- major: 118 документів
```

### Додаткові тести

1. **Перевірка ініціалізації (init_db.py)**:
   - Коректність завантаження даних
   - Створення ембедінгів
   - Збереження у ChromaDB

2. **Верифікація бази (verify_db.py)**:
   - Перевірка цілісності даних
   - Тестування пошуку
   - Валідація метаданих

## API Endpoints

### GET /
Головна сторінка з веб-інтерфейсом.

### POST /api/reading
Отримання читання Таро.

**Request:**
```json
{
    "question": "Ваше питання тут"
}
```

**Response:**
```json
{
    "cards": [
        {
            "name": "The Fool",
            "is_reversed": false,
            "image_path": "/static/images/cards/MajorArcana/0.jpg"
        },
        // ... інші карти
    ],
    "reading": "Текст інтерпретації..."
}
```

## Технічні деталі

### Векторна база даних

- **Тип**: ChromaDB
- **Ембедінги**: SentenceTransformers (all-MiniLM-L6-v2)
- **Структура документів**:
  - 234 документи загалом (78 карт × 3 документи на карту)
  - Для кожної карти створюються:
    1. Повний документ з усією інформацією
    2. Документ для прямого положення
    3. Документ для перевернутого положення
  - Метадані включають:
    - `name`: назва карти
    - `type`: major/minor
    - `suit`: масть або NA для старших арканів
    - `aspect`: upright/reversed (для специфічних документів)
- **Розподіл**:
  - Старші Аркани: 66 документів (22 карти × 3)
  - Молодші Аркани: 168 документів (56 карт × 3)

### LangChain компоненти

1. **ChatOpenAI**
   - Модель: gpt-4-turbo-preview
   - Температура: 0.7
     - Контролює креативність та випадковість відповідей
     - 0.0: максимально детерміновані, консистентні відповіді
     - 1.0: максимально креативні, різноманітні відповіді
     - 0.7: збалансоване значення для тарологічних інтерпретацій
       - Достатньо креативності для унікальних читань
       - Зберігає зв'язок з традиційними значеннями карт
       - Дозволяє адаптувати інтерпретації під контекст питання
       - Забезпечує різноманітність відповідей при повторних запитах

2. **Retrieval Chain**
   - Тип пошуку: similarity
   - K найближчих сусідів: 5

3. **Prompt Template**
   - Системний промпт для таролога
   - Контекстуальна інтеграція карт
   - Українська мова відповідей

### Frontend особливості

- Адаптивний дизайн
- Розміри карт:
  - Десктоп: 300px
  - Планшет: 240px
  - Мобільний: 180px
- Автоматичне відображення перевернутих карт
- Шрифти: Playfair Display, Cormorant Garamond

## Обмеження та особливості

1. **Векторна база**
   - Зберігається локально
   - Не включена в git через розмір
   - **Потребує ініціалізації при першому запуску**

2. **API ключі**
   - Потрібен OpenAI API ключ
   - Зберігається в .env файлі

3. **Ресурси**
   - Потребує ~1GB для векторної бази
   - Використовує GPU для ембедінгів (якщо доступно)
   - Асинхронна обробка запитів
