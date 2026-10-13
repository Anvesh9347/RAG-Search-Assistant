"""Langgraph nodes for Rag Workflow"""

from src.state.rag_state import Rag_state
class RAGNodes:
    """contains node functions for RAG workflow"""

    def __init__(self,retriever,llm)
        """Intialize ther Rag nodes
        
        Args:
            Retriever:Documents retriever instance
            llm: language model instance"""

        self.retriever=retriever
        self.llm=llm


    def retriever_docs(self,state:Rag_state)->Rag_state:

        docs=self.retriever.invoke(state:question)
        return  Rag_state(
            question=state.question
            retrived_docs=docs

        )

    def generate_answer(self,state:Rag_state)->Rag_state:
        # combine retrived documents into context
        context="\n\n".join([doc.page_content for doc in state.retrieved_docs])

        # Create prompt
        prompt=f""" Answer the question based on context.
        
    Context:{context}
    Question:{state.question}"""

         #Generate response
        response=self.llm.invoke(prompt)

        return Rag_state(
            question=state.question,
            retrived_docs=state.retrieved_docs,
            answer=response.content
        )