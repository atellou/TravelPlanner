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
from prompt import (
    QUESTION_PLANNER_INSTRUCTION,
    # RESEARCHER_PLAN_INSTRUCTION,
    REACT_REFLECT_PLANNER_INSTRUCTION,
    # REFLECT_INSTRUCTION,
    # RESEARCH_CRITIQUE_PROMPT,
)


class Queries(BaseModel):
    queries: List[str]


class GenerateResponse(BaseModel):
    plan: Optional[str] = Field(description="The formulated plan.", default="")
    action: str = Field(description="The message that should be returned to the user.")
    though: str = Field(
        description="The though process performed to reach the conclusions."
    )
    tools: list = Field(
        description="The list of tools that should be used to answer the question.",
        default=[],
    )


class AgentState(TypedDict):
    message: str
    messages: Annotated[list[AnyMessage], add_messages]
    queries: List[str]
    plan: str
    information: str
    revision_number: int
    max_revisions: int


class AgentPlanner:
    def __init__(self, tools: List[BaseTool], model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.llm = ChatVertexAI(
            model=model_name,
            temperature=0,
            verbose=True,
            max_retries=1,
        )
        # NOTE: Temporal beahviour, the idea is to be flexible with the tools usage
        assert any(
            [isinstance(tool, VectorRetrieverTool) for tool in tools]
        ), "At least one tool should be a VectorRetrieverTool."
        self.tools = ToolNode(tools)
        self.llm_tools = self.llm.bind_tools(tools)
        self.graph = self.define_graph()

    def define_graph(self):
        # Nodes
        builder = StateGraph(AgentState)
        builder.add_node("generate_queries", self.generate_queries_node)
        builder.add_node("query_or_respond", self.query_or_respond_node)
        builder.add_node("tools", self.tools)
        builder.add_node("generate", self.generate_node)
        # builder.add_node("reflect", self.reflection_node)
        # Edges Order
        builder.set_entry_point("generate_queries")
        builder.add_edge("generate_queries", "query_or_respond")
        builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        # builder.add_edge("tools", "reflect")
        builder.add_edge("tools", "generate")
        builder.add_edge("generate", END)

        self.conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        return builder.compile(checkpointer=self.memory)

    def display_graph(self):
        return Image(self.graph.get_graph().draw_png())

    # First Stage
    def generate_queries_node(self, state: AgentState):
        response = self.llm.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=QUESTION_PLANNER_INSTRUCTION),
                HumanMessage(content=state["message"]),
            ]
        )
        return {
            "queries": response.queries,
        }

    def query_or_respond_node(self, state: AgentState):
        prompt = (
            """The following queries about the travel were provided by an expert: {}"""
        )
        responses = []
        for query in state["queries"]:
            time.sleep(60)
            print(query)
            response = self.llm_tools.invoke(
                [
                    SystemMessage(content=prompt.format(query)),
                    HumanMessage(content=state["message"]),
                ]
            )
            responses.append(response)
        print(responses)
        return {
            "messages": responses,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    # Second Stage
    def get_last_tool_messages(self, state: AgentState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        return recent_tool_messages[::-1]

    def generate_node(self, state: AgentState):
        state
        tool_messages = self.get_last_tool_messages(state)
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        message = [
            SystemMessage(
                content=REACT_REFLECT_PLANNER_INSTRUCTION.format(
                    information=docs_content
                )
            ),
            HumanMessage(content=state["message"]),
        ]
        response = self.llm.invoke(message)
        return {
            "messages": [response],
            "plan": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    # Third Stage
    # def reflection_node(self, state: AgentState):
    #     tool_messages = self.get_last_tool_messages(state)
    #     docs_content = "\n\n".join(doc.content for doc in tool_messages)
    #     message = [
    #         SystemMessage(
    #             content=REFLECT_INSTRUCTION.format(
    #                 information=docs_content
    #             )
    #         ),
    #         HumanMessage(content=state["message"]),
    #     ]
    #     response = self.llm.invoke(message)
    #     return {
    #         "messages": [response],
    #         "reflection": response.content,
    #     }
