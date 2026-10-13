RAG Assistant with External Knowledge

A Retrieval-Augmented Generation (RAG) assistant built with LangChain and LangGraph and deployed on Streamlit. It answers questions from your own sources (web URLs, PDFs, text files) and uses Wikipedia for general knowledge.

What I built
Document loading and chunking (document_processor.py): loads content from URLs, PDF folders and text files, then splits it into chunks (size 500, overlap 250) with RecursiveCharacterTextSplitter.
Vector store (vectorstore.py): embeds the chunks with OpenAI embeddings, stores them in FAISS, and provides a retriever for similarity search.
Workflow state (rag_state.py): a Pydantic Rag_state carries the question, retrieved documents and final answer through the workflow.
Nodes and agent (React.py): a retriever node fetches relevant chunks, and a responder node runs a ReAct agent with two tools, a document retriever and Wikipedia.
Graph (graph_builder.py): a LangGraph workflow retriever → responder → END, run with run(question).
Deployment: the app runs on Streamlit and answers based on the files you provide.
