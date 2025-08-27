"""
Module for managing the vector store for tarot card data
"""
import os
from typing import List, Dict

# Відключаємо телеметрію ChromaDB перед імпортом
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .chromadb_config import get_chromadb_settings

class TarotVectorStore:
    """Class for managing the vector store for tarot card data"""
    
    def __init__(self, persist_directory: str):
        """Initialize vector store"""
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Configure ChromaDB client with disabled telemetry
        self.client_settings = get_chromadb_settings()
        
        # Create or load the vector store
        if os.path.exists(persist_directory):
            self.db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_metadata={"hnsw:space": "cosine"},
                client_settings=self.client_settings
            )
        else:
            self.db = None

    def create_or_update(self, documents: List[Dict[str, str]]) -> None:
        """Create or update the vector store with new documents"""
        # Convert to LangChain documents
        langchain_docs = [
            Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            ) for doc in documents
        ]
        
        # Split documents
        split_docs = self.text_splitter.split_documents(langchain_docs)
        
        # Create new vector store with persistence
        self.db = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": "cosine"},  # використовуємо косинусну відстань для порівняння
            client_settings=self.client_settings
        )

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        if not self.db:
            raise ValueError("Vector store not initialized")
        return self.db.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4):
        """Perform similarity search with relevance scores"""
        if not self.db:
            raise ValueError("Vector store not initialized")
        return self.db.similarity_search_with_score(query, k=k)
