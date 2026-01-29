from langgraph.graph import StateGraph,START,END
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
import os

from typing import TypedDict,Literal,Annotated
from dotenv import load_dotenv
from amadeus import Client, ResponseError

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from ddgs import DDGS

import requests
import random

load_dotenv()




eval_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="conversation"

)
gen_llm =HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
gen_model = ChatHuggingFace(llm=gen_llm)

model = ChatHuggingFace(llm=eval_llm)


class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage],add_messages]


# Tools

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo."""
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        return str(results)

@tool
def calculator(first_num: float, second_num:float,operation:str) ->dict:
    """Perform a basic arithmetic operation on two number.
     Supported operation: add,sub,mul,div """
    
    try:
        if operation == "add":
            result=first_num + second_num
        elif operation == "sub":
            result=first_num - second_num
        elif operation == 'mul':
            result=first_num * second_num
        elif operation == 'div':
            if second_num ==0:
                return {"error":"Division by zero is not allowed"}
            result=first_num / second_num
        else:
            return {"error":f"Unsupported operation {operation}"}
        
     
        return {"first_num":first_num,"second_num":second_num,"operation":operation,"result":result}
    
    except Exception as e:
        return {"error":str(e)}

@tool
def get_stock_price(symbol:str) -> dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL','TSLA) using Alpha Vantage with API key in the URL. """
    
    url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=RG92YAWBNFARKP1M"
    r= requests.get(url)
    return r.json()


def chat_node(state:ChatState) -> ChatState:
    messages = state['messages']
    response=model_tool.invoke(messages)

    return {'messages':[response]}

conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

# Make tool list

tools = [get_stock_price,search_tool,calculator]

# make the llm tools-aware
model_tool=gen_model.bind_tools(tools)


tool_node = ToolNode(tools)



# checkpointer
checkpointer = SqliteSaver(conn=conn)

# graph structur

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)


# add edges
graph.add_edge(START,"chat_node")


# if the LLM asked for a tool, go to ToolNode: else finish
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools','chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)


# test part

def rec_all_thread():
    all_thread=set()
    for checkpoint in checkpointer.list(None):
     all_thread.add(checkpoint.config['configurable']['thread_id'])
    
    return list(all_thread)



