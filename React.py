"""Langgraph for RAG workflow +React agent inside generate content"""
from typing import List,Optional
from src.state.rag_state import Rag_state
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


#wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNODES:

    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm
        self._agent=None

    def retrieve_docs(self,state:Rag_state)->Rag_state:
        """classic retriever node"""
        docs=self.retriever.invoke(state.question)
        return Rag_state(
            question=state.question,
            retrieved_docs=docs
        )

## Build tools
def _build_tools(self)->List[Tool]:
    """"Build retrirver + wikipedia tools"""
    def retriever_tool_fn(query:str)->str:
        docs:List[Document]=self.retriever.invoke(query)
        if not docs:
            return "No document found"
        merged=[]
        for i,d in enumerate(docs[:8],start=1):
            meta=d.metadata if hasattr(d,'metadata')else{}
            title=meta.get('title') or meta.get('source') or f'doc_{i}'
            merged.append(f'[{i}] {title}\n{d.page_content}')
        return "\n\n".join(merged)

    retriever_tool=Tool(
        name='retriever',
        description="FETCH passage from indexed vectorstore",
        func=retriever_tool_fn
    )
    wiki=WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=3,lang="en")
    )
    wikipedia_tool=Tool(
        name='wikipedia',
        description='search wikipedia for general knowledge',
        func=wiki.run,
    )
    return[retriever_tool,wikipedia_tool]


## Build Agent

def _build_agent(self):
    """React agent with tools"""
    tools=self._build_tools()
    system_prompt=("You are a helpful RAG agent",
                   "prefer 'retriever' for user provided docs;use 'wikipedia' for general knowledge",
                   "Return only the final useful answer"
                  )
    self._agent=create_react_agent(self.llm,tools=tools,prompt=system_prompt)


def generate_answer(self,state:Rag_state)->Rag_state:
    """"Generate answer using react agent with wikipedia+retriever tool"""
    if self._agent is None:
        self._build_agent()
    result=self._agent.invoke({"message":[HumanMessage(content=state.question)]})

    messages=result.get("messages",[])
    answer:Optional[str]=None
    if messages:
        answer_msg=messages[-1]
        answer=getattr(answer_msg,"content",None)

    return Rag_state(
        question=state.question,
        retrieved_docs=state.retrieved_docs,
        answer=answer or "could not generate answer"
    )