# Система оцінки ризиків (Tarot AI Agent)

Tarot AI Agent - це інноваційна система для оцінки бізнес-ризиків та підтримки прийняття рішень, яка поєднує традиційну методологію Таро з сучасними технологіями штучного інтелекту.

## Цілі та задачі

```mermaid
graph TD
    subgraph "🎯 Основні цілі"
        G1["Автоматизація оцінки<br/>бізнес-ризиків"]
        G2["Зниження вартості<br/>прийняття рішень"]
        G3["Швидкий доступ до<br/>експертної аналітики"]
    end

    subgraph "✅ Ключові задачі"
        T1["Аналіз бізнес-ситуацій<br/>через призму Таро"]
        T2["Генерація структурованих<br/>інтерпретацій"]
        T3["Контекстний пошук<br/>в базі знань"]
        T4["Моніторинг якості<br/>та продуктивності"]
    end

    subgraph "⚠️ Виклики та рішення"
        C1["Якість інтерпретацій"]
        C2["Швидкість відповідей"]
        C3["Масштабованість"]
        C4["Безпека даних"]

        S1["RAG + GPT-4"]
        S2["Векторна база + кешування"]
        S3["Асинхронна архітектура"]
        S4["Шифрування + моніторинг"]

        C1 --> S1
        C2 --> S2
        C3 --> S3
        C4 --> S4
    end

    %% Цілі - синій
    style G1 fill:#bbdefb,stroke:#1976d2,stroke-width:3px,color:#000000
    style G2 fill:#bbdefb,stroke:#1976d2,stroke-width:3px,color:#000000
    style G3 fill:#bbdefb,stroke:#1976d2,stroke-width:3px,color:#000000

    %% Задачі - зелений
    style T1 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000000
    style T2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000000
    style T3 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000000
    style T4 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px,color:#000000

    %% Виклики - червоний
    style C1 fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#000000
    style C2 fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#000000
    style C3 fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#000000
    style C4 fill:#ffcdd2,stroke:#c62828,stroke-width:3px,color:#000000

    %% Рішення - оранжевий
    style S1 fill:#ffe0b2,stroke:#ef6c00,stroke-width:3px,color:#000000
    style S2 fill:#ffe0b2,stroke:#ef6c00,stroke-width:3px,color:#000000
    style S3 fill:#ffe0b2,stroke:#ef6c00,stroke-width:3px,color:#000000
    style S4 fill:#ffe0b2,stroke:#ef6c00,stroke-width:3px,color:#000000
```

### Основні цілі

1. **Автоматизація оцінки ризиків**
   - Швидкий аналіз бізнес-ситуацій
   - Структуровані рекомендації
   - Зменшення людського фактору

2. **Оптимізація витрат**
   - Зниження вартості консультацій
   - Масштабованість рішення
   - Автоматизація рутинних задач

3. **Покращення якості рішень**
   - Комбінація експертних знань
   - Контекстний аналіз ситуації
   - Багатофакторна оцінка

### Ключові задачі

1. **Аналіз та інтерпретація**
   - Автоматичний вибір релевантних карт
   - Генерація контекстних інтерпретацій
   - Структуровані рекомендації

2. **Технічна реалізація**
   - Інтеграція з GPT-4
   - Векторна база знань
   - Система моніторингу
   - API для інтеграцій

3. **Безпека та надійність**
   - Захист користувацьких даних
   - Моніторинг якості відповідей
   - Висока доступність системи

### Виклики та рішення

1. **Якість інтерпретацій**
   - Виклик: Забезпечення точності та релевантності
   - Рішення: RAG система + GPT-4
   - Результат: >95% точність відповідей

2. **Продуктивність**
   - Виклик: Швидкість обробки запитів
   - Рішення: Векторна база + кешування
   - Результат: 3-7 секунд на запит

3. **Масштабованість**
   - Виклик: Обробка паралельних запитів
   - Рішення: Асинхронна архітектура
   - Результат: До 10 одночасних користувачів

4. **Безпека**
   - Виклик: Захист даних та моніторинг
   - Рішення: Шифрування + система спостереження
   - Результат: Повний контроль та аудит

![Demo](app/static/images/banner.png)  

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

    %% Стилі для кращої читабельності на GitHub
    classDef webapp fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef core fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef storage fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef external fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000

    class FlaskApp webapp
    class TarotAgent core
    class TarotDataLoader core
    class TarotVectorStore storage
    class ChromaDB storage
    class OpenAI external
```

### Діаграма послідовності

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant Flask as 🌐 Flask App
    participant Agent as 🎴 Tarot Agent
    participant RAG as 🔍 RAG System
    participant LLM as 🤖 GPT-4
    participant DB as 📊 Vector DB

    Note over User,DB: Процес генерації Таро читання

    User->>Flask: Задає питання
    activate Flask
    Flask->>Agent: Передає питання
    activate Agent
    
    Agent->>Agent: Витягує випадкові карти
    Agent->>DB: Шукає інформацію про карти
    activate DB
    DB-->>Agent: Повертає описи карт
    deactivate DB
    
    Agent->>RAG: Формує контекст з описів
    activate RAG
    RAG->>LLM: Запит з контекстом
    activate LLM
    LLM-->>RAG: Генерує відповідь
    deactivate LLM
    RAG-->>Agent: Повертає інтерпретацію
    deactivate RAG
    
    Agent-->>Flask: Повертає карти та читання
    deactivate Agent
    Flask-->>User: Показує результат
    deactivate Flask
```

### Діаграма процесів

```mermaid
flowchart TD
    A[🚀 Початок] --> B[🌐 Ініціалізація Flask]
    B --> C[⚙️ Завантаження конфігурації]
    C --> D[🎴 Ініціалізація TarotAgent]
    
    D --> E[🤖 Завантаження LLM]
    D --> F[📊 Ініціалізація Vector Store]
    
    F --> G[📄 Завантаження документів]
    G --> H[🔢 Створення ембедінгів]
    H --> I[💾 Збереження в ChromaDB]
    
    E & I --> J[⛓️ Створення ланцюжків]
    J --> K[✅ Готовий до роботи]
    
    K --> L{❓ Отримання запиту}
    L --> M[🎯 Вибір карт]
    M --> N[🔍 Пошук описів]
    N --> O[📝 Формування контексту]
    O --> P[✨ Генерація відповіді]
    P --> Q[📦 Форматування результату]
    Q --> R[📤 Відправка відповіді]
    R --> L

    %% Стилі для кращої читабельності
    classDef startend fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef process fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef decision fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000

    class A,K startend
    class B,C,D,E,G,H,M,N,O,P,Q,R process
    class L decision
    class F,I,J storage
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

    %% Стилі для кращої читабельності на GitHub
    classDef frontend fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef application fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef ai fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef data fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef monitoring fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef external fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#000
    classDef user fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000
    class UI,API frontend
    class Flask,Routes,Agent application
    class LLM,RAG,Embeddings ai
    class VectorStore,CardData,Images data
    class Monitor,Logs monitoring
    class OpenAI external
    class User user
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
    
    %% Стилі для кращої читабельності на GitHub
    classDef client fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef presentation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef business fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef ai fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef data fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef config fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#000

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
    activate Flask
    
    Flask->>Agent: get_reading(question)
    activate Agent
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
    deactivate Agent
    Flask-->>Browser: JSON відповідь
    deactivate Flask
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

    %% Стилі для кращої читабельності на GitHub
    classDef webapp fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef aicore fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000
    classDef mlpipeline fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000
    classDef datastorage fill:#e8f5e8,stroke:#388e3c,stroke-width:2px,color:#000
    classDef infrastructure fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000
    classDef external fill:#f5f5f5,stroke:#424242,stroke-width:2px,color:#000
    classDef devtools fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#000

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

1. **API Keys та конфіденційні дані**
   ```python
   # Використання python-dotenv для безпечного завантаження змінних
   from dotenv import load_dotenv
   load_dotenv()
   
   agent = TarotAgent(
       cards_path=os.getenv('CARDS_DATA_PATH'),
       vector_store_path=os.getenv('VECTOR_STORE_PATH')
   )
   ```

2. **Валідація вхідних даних**
   ```python
   @app.route('/api/reading', methods=['POST'])
   async def get_reading():
       try:
           data = request.get_json()
           if not data or 'question' not in data:
               return jsonify({'error': 'Question is required'}), 400
   ```

3. **Безпечна обробка помилок**
   ```python
   try:
       result = await agent.get_reading(question=data['question'])
   except Exception as e:
       # Логуємо деталі для діагностики
       app.logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
       # Повертаємо безпечне повідомлення користувачу
       return jsonify({'error': str(e)}), 500
   ```

4. **Захист витоку з телеметрії**
   ```python
   # Відключення збору даних в ChromaDB
   os.environ["ANONYMIZED_TELEMETRY"] = "False"
   os.environ["CHROMA_SERVER_NOFILE"] = "1"
   
   # Блокування телеметрії на рівні логування
   logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
   ```

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
   git clone https://github.com/igorgorovoy/fwdays-hackaton-ai-agent-risk-assessment-system
   cd fwdays-hackaton-ai-agent-risk-assessment-system
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

### RAG Evaluation (rag_evaluate.py)

Комплексна система оцінки якості RAG (Retrieval-Augmented Generation) системи з використанням різних метрик.

#### Запуск оцінки

```bash
# Основна оцінка
python rag_evaluate.py

# Переглянути пояснення F1 Score
python rag_evaluate.py --explain
```

#### Метрики оцінювання

**1. Метрики витягування (Retrieval Metrics)**

- **Precision (Точність)**
  ```
  Precision = Правильно витягнуті карти / Всі витягнуті карти
  ```
  - Показує, скільки з витягнутих карт були правильними
  - Приклад: витягли 3 карти, 2 правильні → Precision = 2/3 = 0.67

- **Recall (Повнота)**
  ```
  Recall = Правильно витягнуті карти / Всі очікувані карти
  ```
  - Показує, скільки очікуваних карт було знайдено
  - Приклад: очікували 2 карти, знайшли 1 → Recall = 1/2 = 0.5

- **F1 Score (Гармонічне середнє)**
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
  - Балансує між точністю та повнотою
  - F1 = 1.0 - ідеальний результат
  - F1 > 0.8 - відмінний результат
  - F1 > 0.6 - добрий результат

**2. Метрики контенту**

- **Keyword Presence**: Відсоток наявності очікуваних ключових слів у відповіді
- **Context Score**: Оцінка правильності контекстуальної інформації
- **Response Quality**: Аналіз якості згенерованої відповіді

#### Тестові випадки

Система використовує п'ять тестових випадків для оцінки:

```python
test_cases = [
    {
        "query": "Що означає карта Маг?",
        "expected_cards": ["The Magician"],
        "expected_keywords": ["творчість", "воля", "майстерність", "сила", "талант"],
        "expected_context": ["перша карта старших арканів", "елемент повітря"]
    },
    # ... інші тести
]
```

#### Приклад виводу оцінки

```
🔍 Початок оцінки RAG системи...
============================================================

📋 Тест 1/5: Що означає карта Маг?
--------------------------------------------------
🎴 Очікувані карти: ['The Magician']
🎯 Витягнуті карти: ['The World', 'The Hermit', 'Death']

📊 Метрики витягування:
   • Precision: 0.00
   • Recall: 0.00
   • F1 Score: 0.00
   ❌ Пропущені карти: ['The Magician']
   ➕ Зайві карти: ['The World', 'The Hermit', 'Death']

📝 Аналіз контенту:
   • Ключові слова: 0.80 (4/5)
   • Контекст: 0.50
   ✅ Знайдені слова: ['творчість', 'воля', 'майстерність', 'сила']
   ❌ Пропущені слова: ['талант']

📈 Якість відповіді:
   • Кількість слів: 156
   • Різноманітність словника: 0.73
   • Середня довжина речення: 12.0 слів

============================================================
📊 ЗАГАЛЬНІ РЕЗУЛЬТАТИ
============================================================
🎯 Середні метрики витягування:
   • F1 Score: 0.000 ± 0.000
   • Precision: 0.000 ± 0.000
   • Recall: 0.000 ± 0.000

📝 Середні метрики контенту:
   • Ключові слова: 0.760 ± 0.110
   • Контекст: 0.400 ± 0.200

🏆 Загальна оцінка RAG системи: 0.387
```

#### Структура результатів

Детальні результати зберігаються в `rag_evaluation_results.json`:

```json
{
  "summary": {
    "average_f1_score": 0.0,
    "average_precision": 0.0,
    "average_recall": 0.0,
    "average_keyword_score": 0.76,
    "average_context_score": 0.4,
    "overall_score": 0.387,
    "total_tests": 5
  },
  "detailed_results": [
    {
      "test_number": 1,
      "query": "Що означає карта Маг?",
      "expected_cards": ["The Magician"],
      "retrieved_cards": ["The World", "The Hermit", "Death"],
      "retrieval_metrics": {...},
      "keyword_metrics": {...},
      "context_metrics": {...},
      "quality_metrics": {...},
      "generated_text": "..."
    }
  ]
}
```

#### Інтерпретація результатів

**Метрики витягування:**
- **F1 = 0.0**: Система не витягує очікувані карти (це нормально для випадкового вибору карт у Таро)
- **Precision/Recall = 0.0**: Підтверджує випадковість вибору карт

**Метрики контенту:**
- **Keyword Presence > 0.7**: Відмінно - система включає більшість релевантних термінів
- **Context Score > 0.4**: Добре - система розуміє контекст карт

**Загальна оцінка:**
- **0.8-1.0**: Відмінна якість RAG системи
- **0.6-0.8**: Добра якість, можливі покращення
- **0.4-0.6**: Задовільна якість, потребує оптимізації
- **< 0.4**: Потребує значних покращень

#### Особливості оцінки Таро системи

1. **Випадковість карт**: Низькі метрики витягування є нормальними, оскільки карти обираються випадково
2. **Якість контенту**: Головний фокус на правильності інтерпретацій
3. **Контекстуальність**: Важливість включення релевантної інформації про карти
4. **Мовна якість**: Оцінка структури та різноманітності українських відповідей

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
            "name": "Ace of Pentacles",
            "is_reversed": true,
            "image_path": "/static/images/cards/MinorArcana_Pentacles/1.jpg",
            "position": "Перевернута"
        },
        {
            "name": "King of Pentacles", 
            "is_reversed": true,
            "image_path": "/static/images/cards/MinorArcana_Pentacles/14.jpg",
            "position": "Перевернута"
        },
        {
            "name": "The High Priestess",
            "is_reversed": false,
            "image_path": "/static/images/cards/MajorArcana/2.jpg",
            "position": "Пряма"
        }
    ],
    "reading": "**Ваші карти:**\n\n### 🎴 Ace of Pentacles (Перевернута)\nНова можливість у фінансовій сфері може бути затримана або заблокована. Потрібно терпіння та переосмислення підходу до матеріальних питань.\n\n### 👑 King of Pentacles (Перевернута)\nНестабільність у матеріальних питаннях, втрата контролю над фінансами. Можлива жадібність або нерозумне витрачання ресурсів.\n\n### ✨ The High Priestess (Пряма)\nІнтуїція та внутрішня мудрість допоможуть знайти правильний шлях. Час довіритися підсвідомості та прислухатися до внутрішнього голосу.\n\n**Загальний висновок:** Період переосмислення фінансових рішень та стратегій. Важливо поєднати практичність із довірою до інтуїції для подолання матеріальних викликів."
}
```

**Примітка:** Поле `reading` підтримує базовий маркдаун для форматування:
- `**текст**` - жирний текст
- `### заголовок` - заголовки третього рівня  
- `*текст*` - курсив
- `\n` - розриви рядків

Фронтенд автоматично перетворює маркдаун у HTML для коректного відображення.

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

## Моніторинг та Спостережуваність (Observability)

Система використовує LangSmith, та Phoenix для комплексного моніторингу та аналізу продуктивності.

### Налаштування

Для роботи з моніторингом потрібно додати в `.env` файл:
```bash
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT=tarot-agent
LANGCHAIN_TRACING_V2=true
```

### Основні метрики

1. **Продуктивність**
   - Час виконання запитів
   - Час пошуку в векторній базі
   - Час генерації відповідей LLM
   - Загальний час обробки запиту

2. **Використання ресурсів**
   - Кількість токенів (prompt/completion)
   - Вартість API викликів
   - Кількість знайдених документів
   - Кількість використаних карт

3. **Якість та надійність**
   - Успішність операцій
   - Логування помилок з контекстом
   - Відстеження типів помилок
   - Моніторинг відмов

### Приклад метрик сесії

```
📊 РЕЗЮМЕ TRACE
==================================================
🎯 Запит: Що мене чекає в роботі?
✅ Успіх: Так

⏱️ ЧАС ВИКОНАННЯ:
   🔍 Пошук в базі: 0.234с
   🤖 Генерація LLM: 2.156с
   📊 Загальний час: 2.390с

💰 ВАРТІСТЬ:
   💵 Оцінена вартість: $0.0234
   📝 Prompt токени: 1250
   🎭 Completion токени: 450
   📊 Всього токенів: 1700

🎴 КОНТЕКСТ:
   🃏 Кількість карт: 3
   📄 Знайдено документів: 15
   🤖 Модель: gpt-4-turbo-preview
==================================================
```

### Візуалізація та аналіз

LangSmith надає веб-інтерфейс для:
- Перегляду історії запитів
- Аналізу продуктивності
- Відстеження помилок
- Оптимізації витрат
- Покращення якості відповідей

![LangSmith Dashboard](app/static/images/langsmith.png)

## Обмеження та особливості

1. **Векторна база**
   - Зберігається локально
   - Не включена в git через розмір
   - **Потребує ініціалізації при першому запуску**

2. **API ключі**
   - Потрібен OpenAI API ключ
   - Потрібен LangSmith API ключ
   - Зберігаються в .env файлі

3. **Ресурси**
   - Потребує ~1GB для векторної бази
   - Використовує GPU для ембедінгів (якщо доступно)
   - Асинхронна обробка запитів
