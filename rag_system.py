
import os
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever


class MyFirstRag:
    def __init__(
        self, llm_provider, model_name, embedding_model, temperature, document_path: str = None, vector_store_path: str = "faiss_index_openai",
       
    ):  
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.temp = temperature
        self.document_path = document_path
        self.vector_store_path = vector_store_path
        self.documents = []  # Store document chunks for BM25

        # Get API keys from environment variables
        open_ai_api = os.getenv("OPENAI_API_KEY")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        nvidia_api = os.getenv("NVIDIA_API_KEY")

        if self.llm_provider == 'openai':
            
            self.llm = ChatOpenAI(
                model = self.model_name,
                temperature = self.temp,
                api_key = open_ai_api,
            )
        
        else:
            self.llm = ChatOpenAI(
                api_key = deepseek_api_key, 
                model = 'deepseek-chat',
                base_url = 'https://api.deepseek.com',
                streaming = True
            )

        self.embeddings = OpenAIEmbeddings(
            api_key = open_ai_api, model = self.embedding_model
        )

    
    #============================================================
    # CREATE RETRIEVERS (BM25 + SEMANTIC SEARCH)
    #============================================================

    def _setup_retrievers(self):
        """Load existing or create new vector store and BM25 index"""
        try:
            # Check if we have existing indices
            faiss_exists = os.path.exists(f"{self.vector_store_path}/index.faiss")
            bm25_exists = os.path.exists(f"{self.vector_store_path}/bm25_docs.json")

            if faiss_exists and bm25_exists:
                print("📂 Loading existing vector store and BM25 index...")
                # Load FAISS
                faiss_retriever = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                ).as_retriever(search_kwargs={"k": 5})

                # Load BM25 documents
                with open(
                    f"{self.vector_store_path}/bm25_docs.json", "r", encoding="utf-8"
                ) as f:
                    docs_data = json.load(f)
                    self.documents = [Document(**doc) for doc in docs_data]

                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(self.documents, k=5)

            elif self.document_path:
                print("Creating new vector store and BM25 index...")
                # Load and split documents
                with open(self.document_path, "r", encoding="utf-8") as f:
                    doc_content = f.read()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1000, chunk_overlap=200, length_function=len
                )

                self.documents = text_splitter.create_documents([doc_content])

                # Create FAISS vector store
                faiss_vectorstore = FAISS.from_documents(
                    self.documents, self.embeddings
                )
                faiss_vectorstore.save_local(self.vector_store_path)
                faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

                # Create BM25 retriever
                bm25_retriever = BM25Retriever.from_documents(self.documents, k=5)

                # Save BM25 documents for future use
                docs_data = [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in self.documents
                ]

                with open(
                    f"{self.vector_store_path}/bm25_docs.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(docs_data, f, ensure_ascii=False, indent=2)

                print(
                    f"Vector store and BM25 index saved to {self.vector_store_path}"
                )
            else:
                raise ValueError(
                    "No document path provided and no existing indices found"
                )

            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
            )

            return ensemble_retriever

        except Exception as e:
            print(f"Error setting up retrievers: {e}")
            raise
    
    #===================================================
    # RERANKING WITH NVIDIA API (OPTIONAL)
    #====================================================
    def _reranked_retriever(self):
        """rerank the retriever for better filtering"""
        
        ens_rtr = self._setup_retrievers()
        
        # Get NVIDIA API key from environment
        nvidia_api = os.getenv("NVIDIA_API_KEY")
        
        reranker = NVIDIARerank(
            model = "nv-rerank-qa-mistral-4b:1", 
            api_key=nvidia_api,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=ens_rtr
        )
        return compression_retriever
    
    #===========================================================
    # RAG CHAIN
    #===========================================================

    def create_rag_chain(self, query, rerank=False):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                    """You are an expert assistant. Answer the question based ONLY on the following context. 
                If the context doesn't contain the answer, say "I cannot find this information in the document."

                Instructions:
                1. Answer directly and concisely
                2. Use only information from the context
                3. If unsure, say you don't know
                4. Do not make up information

                Context:
                {context}""",
                ),
                ("human", "{input}"),
            ]
        )

        if rerank:
            retriever = self._reranked_retriever()
        else:
            print('Using standard retrievers (no reranking)')
            retriever = self._setup_retrievers()

        # Create the chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        
        if self.llm_provider == 'deepseek':
            return chain
        
        else:
            result = chain.invoke({"input": query})
            return result.get('answer')