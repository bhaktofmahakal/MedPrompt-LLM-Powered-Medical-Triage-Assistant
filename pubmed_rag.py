"""
PubMed RAG Pipeline for Medical AI Symptom Checker
This module implements a Retrieval-Augmented Generation (RAG) pipeline
using PubMed data to enhance the symptom analysis capabilities.
"""

import os
from typing import List, Dict, Any

# Try to import Bio, but provide a fallback if not available
try:
    from Bio import Entrez
    # Set your email for Entrez (required by NCBI)
    Entrez.email = "medprompt@example.com"  # Application email for NCBI API
    BIO_AVAILABLE = True
except ImportError:
    print("Warning: BioPython (Bio) module not found. Using mock PubMed data.")
    BIO_AVAILABLE = False

# Try to import LangChain dependencies
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.schema.document import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LangChain dependencies not found: {str(e)}. Using mock data.")
    LANGCHAIN_AVAILABLE = False

class PubMedRAG:
    """
    A class to implement RAG pipeline using PubMed data
    """
    
    def __init__(self, cache_dir: str = "pubmed_cache"):
        """
        Initialize the PubMed RAG pipeline
        
        Args:
            cache_dir: Directory to cache downloaded PubMed articles
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check for required dependencies
        if not BIO_AVAILABLE:
            print("ERROR: BioPython (Bio) module is required but not found.")
            print("Please install it with: pip install biopython")
            raise ImportError("BioPython (Bio) module is required")
            
        if not LANGCHAIN_AVAILABLE:
            print("ERROR: LangChain dependencies are required but not found.")
            print("Please install them with: pip install langchain langchain_community langchain_huggingface faiss-cpu")
            raise ImportError("LangChain dependencies are required")
            
        self.mock_mode = False
        
        try:
            # Initialize embeddings model
            print("Initializing HuggingFaceEmbeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize text splitter
            print("Initializing RecursiveCharacterTextSplitter...")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Initialize vector store
            self.vector_store = None
            print("RAG pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing RAG pipeline: {str(e)}")
            raise RuntimeError(f"Failed to initialize RAG pipeline: {str(e)}")
    
    def search_pubmed(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search PubMed for articles related to the query
        
        Args:
            query: Search query for PubMed
            max_results: Maximum number of results to return
            
        Returns:
            List of PubMed IDs
        """
        try:
            print(f"Searching PubMed for: {query}")
            # Search PubMed
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            # Get the ID list
            id_list = search_results.get("IdList", [])
            
            if not id_list:
                print(f"No PubMed articles found for query: {query}")
                # Try a more general search if specific query returns no results
                general_terms = query.split()[:3]  # Use first few words for a more general search
                general_query = " OR ".join(general_terms)
                print(f"Trying more general search: {general_query}")
                
                search_handle = Entrez.esearch(db="pubmed", term=general_query, retmax=max_results)
                search_results = Entrez.read(search_handle)
                search_handle.close()
                
                id_list = search_results.get("IdList", [])
                
            print(f"Found {len(id_list)} PubMed articles")
            return id_list
        except Exception as e:
            print(f"Error searching PubMed: {str(e)}")
            raise RuntimeError(f"Failed to search PubMed: {str(e)}")
    
    def fetch_pubmed_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch PubMed articles by their IDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of article data
        """
        if not pmids:
            print("No PubMed IDs provided to fetch")
            raise ValueError("No PubMed IDs provided to fetch articles")
            
        try:
            print(f"Fetching {len(pmids)} PubMed articles...")
            # Fetch articles
            fetch_handle = Entrez.efetch(db="pubmed", id=pmids, rettype="xml")
            articles = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            results = []
            for article in articles["PubmedArticle"]:
                article_data = {}
                
                # Extract article metadata
                article_data["pmid"] = article["MedlineCitation"]["PMID"]
                article_data["title"] = article["MedlineCitation"]["Article"]["ArticleTitle"]
                
                # Extract abstract if available
                if "Abstract" in article["MedlineCitation"]["Article"]:
                    abstract_parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                    article_data["abstract"] = " ".join([str(part) for part in abstract_parts])
                else:
                    article_data["abstract"] = "No abstract available for this article."
                
                results.append(article_data)
            
            if not results:
                print("No article data could be extracted from PubMed")
                raise ValueError("Failed to extract article data from PubMed")
                
            return results
        except Exception as e:
            print(f"Error fetching PubMed articles: {str(e)}")
            raise RuntimeError(f"Failed to fetch PubMed articles: {str(e)}")
    
    def create_documents_from_articles(self, articles: List[Dict[str, Any]]) -> List:
        """
        Create LangChain documents from PubMed articles
        
        Args:
            articles: List of article data
            
        Returns:
            List of LangChain documents
        """
        if self.mock_mode:
            # Return mock documents (simple dict structure instead of Document objects)
            documents = []
            for article in articles:
                content = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
                metadata = {
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "source": "PubMed"
                }
                documents.append({"page_content": content, "metadata": metadata})
            return documents
            
        documents = []
        for article in articles:
            content = f"Title: {article['title']}\n\nAbstract: {article['abstract']}"
            metadata = {
                "pmid": article["pmid"],
                "title": article["title"],
                "source": "PubMed"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def build_vector_store(self, documents: List):
        """
        Build a vector store from documents
        
        Args:
            documents: List of documents to index
        """
        if self.mock_mode:
            # Skip vector store creation in mock mode
            return
            
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
    
    def query_vector_store(self, query: str, k: int = 3) -> List:
        """
        Query the vector store for relevant documents
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if self.mock_mode:
            # Return the mock documents directly
            mock_docs = self.create_documents_from_articles(self.fetch_pubmed_articles([]))
            return mock_docs[:k]
            
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_vector_store first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def process_symptoms(self, symptoms: str) -> List:
        """
        Process symptoms to retrieve relevant medical information
        
        Args:
            symptoms: User symptoms
            
        Returns:
            List of relevant documents
        """
        try:
            print(f"Processing symptoms: {symptoms}")
            
            # Clean up symptoms text
            clean_symptoms = symptoms.replace("\n", " ").replace("-", " ").strip()
            
            # Search PubMed for relevant articles
            pmids = self.search_pubmed(clean_symptoms, max_results=10)
            
            if not pmids:
                print("No PubMed articles found for the symptoms.")
                raise ValueError("No PubMed articles found for the given symptoms")
            
            # Fetch articles
            articles = self.fetch_pubmed_articles(pmids)
            
            # Create documents
            documents = self.create_documents_from_articles(articles)
            
            # Build vector store
            self.build_vector_store(documents)
            
            # Query vector store
            return self.query_vector_store(clean_symptoms, k=3)
        except Exception as e:
            print(f"Error processing symptoms: {str(e)}")
            raise RuntimeError(f"Failed to process symptoms: {str(e)}")
    
    def get_context_from_symptoms(self, symptoms: str) -> str:
        """
        Get context from symptoms for LLM prompt enhancement
        
        Args:
            symptoms: User symptoms
            
        Returns:
            Context string for LLM prompt
        """
        if not symptoms or len(symptoms.strip()) < 3:
            print("Warning: Symptoms text is too short or empty")
            raise ValueError("Symptom text is too short or empty")
            
        try:
            print(f"Getting context for symptoms: {symptoms}")
            relevant_docs = self.process_symptoms(symptoms)
            
            if not relevant_docs:
                raise ValueError("No relevant medical literature found for these symptoms")
                
            context = "Relevant medical information:\n\n"
            for i, doc in enumerate(relevant_docs, 1):
                if isinstance(doc, dict):
                    # Handle document format if it's a dict
                    content = doc["page_content"]
                else:
                    content = doc.page_content
                context += f"Source {i}:\n{content}\n\n"
            
            return context
        except Exception as e:
            error_msg = str(e)
            print(f"Error retrieving PubMed context: {error_msg}")
            raise RuntimeError(f"Failed to retrieve medical context: {error_msg}")

# Example usage
if __name__ == "__main__":
    rag = PubMedRAG()
    context = rag.get_context_from_symptoms("persistent headache with fever and neck stiffness")
    print(context)