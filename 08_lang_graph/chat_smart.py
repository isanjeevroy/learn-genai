from dotenv import load_dotenv
from pathlib import Path
from typing_extensions import TypedDict
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")

client = OpenAI()

class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]

def chatbot(state: State):
    print("\n\nchabot node ", state)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"user", "content": state.get("user_query")}
        ]
    )

    state["llm_output"] = response.choices[0].message.content
    
    return state

def evaluation_response(state: State) -> Literal["chatbot_gemini", "endnode"]:
    print("\n\nevaluation_response node", state)
    if True:
        return "endnode"
    
    return "chatbot_gemini"

def chatbot_gemini(state: State):
    print("\n\nchatbot_gemini node ", state)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role":"user", "content": state.get("user_query")}
        ]
    )

    state["llm_output"] = response.choices[0].message.content
    
    return state

def endnode(state: State):
    print("\n\nendnode ", state)
    return state

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_gemini", chatbot_gemini)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluation_response)
graph_builder.add_edge("chatbot_gemini", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

updated_state = graph.invoke(State({"user_query":"Hey, What is the 2+2 ?"}))
print("\n\n ", updated_state)