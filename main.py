from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import  add_messages
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
import sqlite3
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# llm + Embedding
llm = ChatGoogleGenerativeAI(model= "gemini-3-flash-preview", google_api_key= GOOGLE_API_KEY)
class CustomGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
embedding = CustomGoogleEmbeddings(
    model="models/gemini-embedding-2"
)


# ingestion Rag
# loading the documents
loader = TextLoader("data.txt", encoding="utf-8") 
docs = loader.load()
print(len(docs))
# splits
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks= splitter.split_documents(docs)
print(len(chunks))
texts = [doc.page_content for doc in chunks]
embeds = embedding.embed_documents(texts)

print("Chunks:", len(texts))
print("Embeddings:", len(embeds))
# embedding
vector_store = FAISS.from_documents(chunks, embedding)
# retriver
retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
# tool define
@tool 
def rag_tool(query: str):
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    return {
        "query":query,
        "context":context
    }

tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)
# state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]
# nodes
def chat_node(state: ChatState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)
# graph
graph= StateGraph(ChatState)
# adding node
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
# edges define
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile()

# test


# Define strict rules for the AI
# system_prompt = """
# You are a helpful assistant. You must answer the user's questions strictly using ONLY the context provided by the 'rag_tool'. 
# If you find the answer in the context, provide it and include a direct quote from the text to prove it.
# If the answer is NOT in the context, you must explicitly say: "I cannot find this information in data.txt."
# Do not use your general knowledge.
# """

# Run the chatbot with the system prompt and the user question
response = chatbot.invoke(
    {"messages": [
        # SystemMessage(content=system_prompt),
        HumanMessage(content="Why we needed the Agentic-ai RAG?")
    ]}
)
# Get only AI message
ai_message = response["messages"][-1]

if isinstance(ai_message.content, list):
    print(ai_message.content[0]["text"])
else:
    print(ai_message.content)