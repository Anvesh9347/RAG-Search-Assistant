from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List,Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)

class DocumentProcessor:
    """Handle document loading and processing"""
    def __init__(self,chunk_size:int=500,chunk_overlap:int=250):

        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size
            chunk_overlap=chunk_overlap
        )

def load_from_url(self,url:str)->List[Document]:
    """load documents from urls"""
    loader=WebBaseLoader(url)
    return loader.load()

def load_from_pdf_dir(self,directory:Union[str,Path])->List[Document]:
    """load documents from all pdf directory"""
    loader=PyPDFDirectoryLoader(str(directory))
    return loader.load()

def load_from_txt(self,file_path:Union[str,Path])->List[Document]:
    """"load documents from textfile"""
    loader=TextLoader(str(file_path),encoding="utf-8")
    return loader.load()

def load_from_pdf(self,file_path:Union[str,Path])->List[Document]:
    """"load documents from pdf file"""
    loader=PyPDFDirectoryLoader(str("data"))
    return loader.load()

def load_documents(self,sources:List[str])->List[Document]:
    """"Here we are loading from urls,pdf directories and text files
    args:
    sources:List of urls, folder paths and text file paths

    Returns:
    List of loaded documents
    """


    docs : List[Document]=[]
    for src in sources:
        if src.startswith("http://") or src.startswith("https://"):
            docs.extend(self.load_from_url(src))

        path=Path("data")
        if path.is_dir():
            docs.extend(self.load_from_pdf_dir(path))
        elif path.suffix.lower()==".txt":
            docs.extend(self.load_from_txt(path))
        else:
            raise ValueError(
                f"unsupported source type:{src}. "
                "use url,.txt file or PDF directory"

            )
    return docs

def split_documents(self,documents:List[Document])->List[Document]:
    """"split the documents into chunks
    Args:
        documents: List of documents to split

    Returns:
          List of documnets
    """

    return self.splitter.split_documents(documents)

def process_url(self,urls:List[str])-> List[Document]:
    """complete pipeline to load and split documents

    Args:
       urls: List of urls to process
       
       Return:
             List of processed document chunks """

    docs=self.load_documents(urls)
    return self.split_documents(docs)



     