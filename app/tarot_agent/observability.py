"""
Module for observability configuration using LangSmith
"""
import os
import time
from typing import Any, Dict, List, Optional
from langsmith import Client
from langchain_core.tracers import LangChainTracer

class TarotObservability:
    """Class for managing observability with LangSmith"""
    
    def __init__(self):
        """Initialize LangSmith client"""
        self.client = Client(
            api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
            api_key=os.getenv("LANGCHAIN_API_KEY")
        )
        self.tracer = LangChainTracer(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent")
        )
    
    def create_trace(self, 
                    question: str,
                    cards: List[Dict[str, Any]],
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new trace for a reading session"""
        try:
            if not os.getenv("LANGCHAIN_API_KEY"):
                print("Warning: LANGCHAIN_API_KEY is not set")
                return None

            extra_data = metadata or {}
            extra_data["session_start_time"] = time.time()
            extra_data["num_cards"] = len(cards)

            run = self.client.create_run(
                project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent"),
                name="tarot_reading",
                run_type="chain",
                inputs={
                    "question": question,
                    "cards": [f"{card['name']} ({'перевернута' if card['is_reversed'] else 'пряма'})" 
                             for card in cards]
                },
                extra=extra_data
            )
            
            if run is None:
                print("Warning: create_run returned None")
                return None
                
            return run.id
        except Exception as e:
            print(f"Error in create_trace: {str(e)}")
            return None
    
    def finalize_trace(self, 
                      trace_id: Optional[str],
                      session_start_time: float,
                      total_cost: float,
                      total_tokens: int,
                      success: bool = True):
        """Завершити trace з фінальними метриками"""
        if trace_id is None:
            return
            
        try:
            session_duration = time.time() - session_start_time
            
            self.client.update_run(
                run_id=trace_id,
                outputs={
                    "success": success,
                    "session_duration_seconds": session_duration,
                    "session_duration_ms": session_duration * 1000,
                    "total_cost_usd": total_cost,
                    "total_tokens_used": total_tokens
                }
            )
        except Exception as e:
            print(f"Error in finalize_trace: {str(e)}")
    
    def log_retrieval(self, 
                     trace_id: Optional[str],
                     query: str,
                     documents: List[str]):
        """Log retrieval step"""
        if trace_id is None:
            return
            
        self.client.create_run(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent"),
            name="retrieval",
            run_type="retriever",
            run_id=trace_id,
            inputs={"query": query},
            outputs={
                "documents": [doc[:200] + "..." for doc in documents]
            }
        )
    
    def log_llm_call(self,
                     trace_id: Optional[str],
                     prompt: str,
                     response: str,
                     execution_time: Optional[float] = None,
                     token_usage: Optional[Dict[str, int]] = None,
                     cost: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log LLM call with performance metrics"""
        if trace_id is None:
            return
        
        extra_data = metadata or {}
        
        # Додаємо метрики продуктивності
        if execution_time is not None:
            extra_data["execution_time_seconds"] = execution_time
            extra_data["execution_time_ms"] = execution_time * 1000
            
        if token_usage:
            extra_data["token_usage"] = token_usage
            extra_data["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
            extra_data["completion_tokens"] = token_usage.get("completion_tokens", 0)
            extra_data["total_tokens"] = token_usage.get("total_tokens", 0)
            
        if cost is not None:
            extra_data["estimated_cost_usd"] = cost
            
        self.client.create_run(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent"),
            name="llm",
            run_type="llm",
            run_id=trace_id,
            inputs={"prompt": prompt},
            outputs={"response": response},
            extra=extra_data
        )
    
    def log_error(self,
                  trace_id: Optional[str],
                  error: Exception,
                  context: Optional[Dict[str, Any]] = None):
        """Log error"""
        if trace_id is None:
            return
            
        self.client.create_run(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent"),
            name="error",
            run_type="tool",
            run_id=trace_id,
            inputs={"error_type": type(error).__name__},
            outputs={
                "error_message": str(error),
                "context": context or {}
            },
            error=str(error)
        )
    
    def calculate_cost(self, token_usage: Dict[str, int], model: str = "gpt-4-turbo-preview") -> float:
        """Обчислити вартість API виклику на основі використання токенів"""
        # Актуальні ціни OpenAI (на лютий 2024)
        pricing = {
            "gpt-4-turbo-preview": {
                "prompt": 0.01 / 1000,  # $0.01 за 1K prompt токенів
                "completion": 0.03 / 1000  # $0.03 за 1K completion токенів
            },
            "gpt-4": {
                "prompt": 0.03 / 1000,
                "completion": 0.06 / 1000
            },
            "gpt-3.5-turbo": {
                "prompt": 0.0015 / 1000,
                "completion": 0.002 / 1000
            }
        }
        
        if model not in pricing:
            model = "gpt-4-turbo-preview"  # За замовчуванням
            
        prompt_cost = token_usage.get("prompt_tokens", 0) * pricing[model]["prompt"]
        completion_cost = token_usage.get("completion_tokens", 0) * pricing[model]["completion"]
        
        return prompt_cost + completion_cost
    
    def start_timer(self) -> float:
        """Почати відлік часу"""
        return time.time()
    
    def end_timer(self, start_time: float) -> float:
        """Закінчити відлік часу та повернути тривалість"""
        return time.time() - start_time
    
    def log_retrieval_with_timing(self, 
                                 trace_id: Optional[str],
                                 query: str,
                                 documents: List[str],
                                 execution_time: float,
                                 num_documents_found: int):
        """Log retrieval step з метриками продуктивності"""
        if trace_id is None:
            return
            
        self.client.create_run(
            project_name=os.getenv("LANGCHAIN_PROJECT", "tarot-agent"),
            name="retrieval",
            run_type="retriever",
            run_id=trace_id,
            inputs={"query": query},
            outputs={
                "documents": [doc[:200] + "..." for doc in documents],
                "num_documents_found": num_documents_found
            },
            extra={
                "execution_time_seconds": execution_time,
                "execution_time_ms": execution_time * 1000,
                "documents_retrieved": len(documents)
            }
        )
    
    def get_tracer(self) -> LangChainTracer:
        """Get LangChain tracer"""
        return self.tracer
    
    def print_trace_summary(self, trace_id: str):
        """Друкує зручне резюме trace з ключовими метриками"""
        try:
            run = self.client.read_run(trace_id)
            extra = run.extra or {}
            
            print("\n" + "="*50)
            print("📊 РЕЗЮМЕ TRACE")
            print("="*50)
            
            # Основна інформація
            print(f"🎯 Запит: {run.inputs.get('question', 'N/A')}")
            print(f"✅ Успіх: {'Так' if extra.get('processing_success', False) else 'Ні'}")
            
            # Метрики продуктивності
            print(f"\n⏱️ ЧАС ВИКОНАННЯ:")
            print(f"   🔍 Пошук в базі: {extra.get('retrieval_time_seconds', 'N/A'):.3f}с")
            print(f"   🤖 Генерація LLM: {extra.get('llm_execution_time_seconds', 'N/A'):.3f}с")
            print(f"   📊 Загальний час: {extra.get('total_execution_time_seconds', 'N/A'):.3f}с")
            
            # Економічні метрики
            print(f"\n💰 ВАРТІСТЬ:")
            print(f"   💵 Оцінена вартість: ${extra.get('estimated_cost_usd', 'N/A'):.4f}")
            print(f"   📝 Prompt токени: {extra.get('prompt_tokens', 'N/A')}")
            print(f"   🎭 Completion токени: {extra.get('completion_tokens', 'N/A')}")
            print(f"   📊 Всього токенів: {extra.get('total_tokens', 'N/A')}")
            
            # Контекст
            print(f"\n🎴 КОНТЕКСТ:")
            print(f"   🃏 Кількість карт: {extra.get('num_cards_drawn', 'N/A')}")
            print(f"   📄 Знайдено документів: {extra.get('documents_retrieved', 'N/A')}")
            print(f"   🤖 Модель: {extra.get('model_used', 'N/A')}")
            
            print("="*50)
            
        except Exception as e:
            print(f"❌ Помилка при читанні trace: {e}")
