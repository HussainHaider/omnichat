from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    

llm = ChatOpenAI(model='gpt-3.5-turbo')

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)

# add notes
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# initial_state = {'messages': [HumanMessage(content="Hello, how are you?")]}
# final_state = chatbot.invoke(initial_state)
# print(final_state['messages'][-1].content)

thread_id = "chat_thread_1"

while True:
    user_input = input("You: ")

    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat.")
        break

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {'messages': [HumanMessage(content=user_input)]}
    final_state = chatbot.invoke(initial_state, config=config)
    print("AI:", final_state['messages'][-1].content)

