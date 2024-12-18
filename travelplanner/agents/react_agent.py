from typing import TypedDict, List, Annotated, Optional
import sqlite3
import time

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

from pydantic import BaseModel, Field
from IPython.display import Image

from tools import VectorRetrieverTool
from prompts.react_agent import REACT_PLANNER_INSTRUCTION


class Queries(BaseModel):
    queries: List[str]


class ReactAgentState(TypedDict):
    message: str
    messages: Annotated[list[AnyMessage], add_messages]
    chat_history: Annotated[list[AnyMessage], add_messages]
    plan: str
    information: str


class ReactAgent:
    def __init__(self, tools: List[BaseTool] = None, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.llm = ChatVertexAI(
            model=model_name,
            temperature=0,
            verbose=True,
            max_retries=1,
        )
        if tools is not None:
            self.tools = ToolNode(tools)
            self.llm_tools = self.llm.bind_tools(tools)
        self.graph = self.define_graph()

    def define_graph(self):
        # Nodes
        builder = StateGraph(ReactAgentState)
        builder.add_node("generate", self.generate_node)
        if getattr(self, "tools", None):
            builder.add_node("tools", self.tools)

        # Edges Order
        builder.set_entry_point("generate")
        if getattr(self, "tools", None):
            builder.add_conditional_edges(
                "generate",
                tools_condition,
                {END: END, "tools": "tools"},
            )
            builder.add_edge("tools", "generate")
        builder.add_edge("generate", END)

        self.conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        return builder.compile(checkpointer=self.memory)

    def display_graph(self):
        return Image(self.graph.get_graph().draw_png())

    # First Stage
    def get_last_tool_messages(self, state: ReactAgentState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        return recent_tool_messages[::-1]

    def generate_node(self, state: ReactAgentState):
        tool_messages = self.get_last_tool_messages(state)
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        state["information"] += f"\n\n{docs_content}" if docs_content else ""
        message = state["chat_history"] + [
            SystemMessage(
                content=REACT_PLANNER_INSTRUCTION.format(information=docs_content)
            ),
            HumanMessage(content=state["message"]),
        ]
        response = self.llm.invoke(message)
        return {
            "messages": [response],
            "chat_history": [HumanMessage(content=state["message"]), response],
            "plan": response.content,
        }
