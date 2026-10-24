from typing import TypedDict, List, Union
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv  # used to store secret stuff like API keys

load_dotenv()
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGroq(model="openai/gpt-oss-20b")

def process(state: AgentState) -> AgentState:
    '''This method will proccess the user input'''
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))

    print(f"AI: {response.content}")

    return state

graph = StateGraph(AgentState)
graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)

agent = graph.compile()

user_input = input("Enter: ")
history = []
while user_input != "exit":
    history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": history})
    history = result['messages']
    user_input = input("Enter: ")