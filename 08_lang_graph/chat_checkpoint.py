from dotenv import load_dotenv
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver
import os

load_dotenv(Path(__file__).parent.parent / ".env")

llm = init_chat_model(
    model="gpt-4.1-mini",
    model_provider="openai"
)

# state = { message: ["Hey, There"] }
# node runs: chatbot(state: ["Hey There"]) => ["Hi, This is a message from ChatBot Node"]
# state = { "messages": ["Hey There", "Hi, This is a message from ChatBot Node"]}
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return { "messages": [response] }


graph_builder = StateGraph(State)

#graph_builder.add_node("name of the node", name)
graph_builder.add_node("chatbot", chatbot)

# (START) -> chatbot -> samplenode -> (END)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Complie the graph
def compile_graph_with_checkpointer(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

MONGODB_URI = os.environ.get("MONGODB_URI")
with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
    graph_with_checkpointer = compile_graph_with_checkpointer(checkpointer=checkpointer)

    config = {
            "configurable": {
                "thread_id": "sanjeev"
            }
        }

    # Invoke the graph ( It's starting from the state only )
    for chunk in graph_with_checkpointer.stream(
        {"messages":["Say me my name and temperature of my city?"]},
        stream_mode="values",
        config=config
        ):
        print(chunk["messages"][-1].content)