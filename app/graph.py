from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv  
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import SystemMessage
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

@tool
def run_command(command: str):
    """
    Takes a command line prompt and executes on the user's machine and returns the 
    output of the command.
    Example: 
    run_command(command="ls") where ls is the command to list the files.
    """
    result = os.system(command = command)
    return result

llm = init_chat_model("openai:gpt-4o")

tools = [run_command]

llm_with_tools = llm.bind_tools(tools=tools)

def chatbot(state: State): # See end of the page for QnA
    system_prompt = SystemMessage( # Write System prompt of Claude (check that github repo)
        content=
        """
        You are an AI assistant who takes an input from user and based on tools you choose the correct tool and execute the commands.
        You can even execute commands and help user with the output of the command. 
        Always make sure to keep your generated code and files in chat_gpt/folder. You can create one if not already there.
        """
    )
    message = llm_with_tools.invoke([system_prompt] + state["messages"])
    assert len(message.tool_calls) <= 1 # Checks that the number of tool calls in the LLM's/Chatbot's response is at most one.
    return {"messages": [message]}

tool_node = ToolNode(tools=tools)

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# Creates a new graph with checkpointer 
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

# What does the assert do? 
# Checks that the number of tool calls in the LLM's response is at most one. LLM tries to call more than one tool in a single response, 
# the program will raise an AssertionError and stop execution at that point. This is a safety check to ensure your workflow only handles one tool call per turn.

# Does this mean you can only use one tool?
# No. You can define and register multiple tools in your tools list. The assert is not about how many tools you have, but about how many tool calls the LLM makes in a single response.

# How does the LLM pick which tool to call?
# The LLM receives the list of available tools (from your tools list). Based on the user's message, it decides which tool (if any) is most appropriate to call.
# It will include a tool call in its response if it thinks a tool should be used. The assert ensures that, even if you have many tools, only one tool call is processed per message.
