from dotenv import load_dotenv
from pathlib import Path
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model

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

def samplenode(state: State):
    print("\n\nInside samplenode node ", state)
    return { "messages": ["Sample Message Appended"] }

graph_builder = StateGraph(State)

#graph_builder.add_node("name of the node", name)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("samplenode", samplenode)

# (START) -> chatbot -> samplenode -> (END)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", "samplenode")
graph_builder.add_edge("samplenode", END)

# Complie the graph
graph = graph_builder.compile()

# Invoke the graph ( It's starting from the state only )
updated_state = graph.invoke({"messages":["Hi, My name is Sanjeev"]})
print("\n\nupdated_state", updated_state)