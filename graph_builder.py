"""Graph builder for langgraph workflow"""

from langgraph.graph import StateGraph,END
from src.state.rag_state import Rag_state
from src.nodes.nodes import RAGNodes

class GraphBuilder:
    """Builds and manages the langgraph workflow"""
    def __init__(self,retriever,llm):
        """initialize the graphbuilder
        Args:
             retriever:Document retriever instance
        llm:langgraph model instance"""

        self.nodes=RAGNodes()
        self.graph=None
    def build(self):
        """Build the Rag workflow graph
        
        Returns:
                compiled graph instance"""

        # create State Graph

        builder=StateGraph(Rag_state)

        # Add notes
        builder.add_node("retriever",self.nodes.retrieve_docs)
        builder.add_node("responder",self.nodes.generate_answer)


        # Set entry point
        builder.set_entry_point("retriever")

        # Add edges
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)

        #Compile Graph
        self.graph=builder.compile()
        return self.graph


    def run(self,question:str) ->dict:

        if self.graph is None:
            self.build()

        initial_state= Rag_state(question=question)
        return self.graph.invoke(initial_state)



