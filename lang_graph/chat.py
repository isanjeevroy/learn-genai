from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langgraph import StateGraph

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    return { "message": ["Hi, This is message from chatboat Node"] }

graph_builderr = StateGraph(State)