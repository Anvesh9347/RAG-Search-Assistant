""""Rag state definition for langgraph"""

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class Rag_state(BaseModel):
    """State object for rag workflow"""
    question:str
    retrieved_docs:List[Document]=[]
    answer:str=""