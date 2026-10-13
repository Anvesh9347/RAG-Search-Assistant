"""Vectorstore for  module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class Vectorstore:
    """Manages vectorstore application"""
    def __init__(self):
        self.embedding=OpenAIEmbeddings
        self.Vectorstore=None
        self.retriever=None

    def create_retriever(self,documents:List[Document]):
    
        """Create vectorestore from documents
        Args:
            List of documents to embeddings
        
        """
        self.Vectorstore=FAISS.from_documents(documents,self.embedding)
        self.retriever=self.Vectorstore.as_retriever()


    def get_retriever(self):
        """get the retriever instance

        Returns:
             retriever instance """
        if self.retriever is None:
            raise ValueError("Vectorstore not intialized. call create vectorstore first")
        return self.retriever
    def retriever(self,query:str,k:int=4) -> List[Document]:
        """"retrieve relevant documents for query
        
        ARGS:
            Query:search query
            k:number of documents to retrieve
            
        Returns: List of documents"""

        if self.retriever is None:
            raise ValueError("vectorstore not initialized.call create vectorstore first")
        return self.retriever . invoke(query)