from typing import TypedDict, List, Annotated, Optional, Literal
import sqlite3
import logging
import ast

logger = logging.getLogger(__name__)

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from pydantic import BaseModel, Field
from IPython.display import Image

from tools import VectorRetrieverTool

from prompts.react_reflection_agent import (
    QUESTION_ANSWERING_INSTRUCTION,
    TASK_DECODER_INSTRUCTION,
    RESEARCH_INSTRUCTION,
)


class AgentState(TypedDict):
    message: str
    task: str
    question: List[str]
    plan: str
    conversation_history: Annotated[list[AnyMessage], add_messages]
    messages: List[str]
    development_concurrent_time: int
    plan: str
    information: Annotated[list[AnyMessage], add_messages]


class ReflectAgent:
    def __init__(self, tools: List[BaseTool] = None, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.llm = ChatVertexAI(
            model=model_name,
            temperature=0,
            verbose=True,
            max_retries=1,
        )
        self.tools = ToolNode(tools)
        self.llm_tools = self.llm.bind_tools(tools)
        self.graph = self.define_graph()

    def define_graph(self):
        # Nodes
        builder = StateGraph(AgentState)
        builder.add_node("question_answer", self.question_answer_node)
        builder.add_node("task_decoder", self.task_decoder_node)
        builder.add_node("researcher", self.researcher_node)
        builder.add_node("tools", self.tools)
        builder.add_node("tool_parser", self.tool_parser_node)
        builder.add_node("the_end", lambda state: {"development_concurrent_time": 0})

        # Edges Order
        builder.add_edge(START, "question_answer")
        builder.add_edge("task_decoder", "researcher")
        builder.add_conditional_edges(
            "researcher",
            tools_condition,
            {END: "question_answer", "tools": "tools"},
        )
        builder.add_edge("tools", "tool_parser")
        builder.add_edge("tool_parser", END)
        builder.add_edge("the_end", END)

        # Checkpointer
        self.conn = sqlite3.connect(
            "checkpoints_reflect.sqlite", check_same_thread=False
        )
        self.memory = SqliteSaver(self.conn)
        return builder.compile(checkpointer=self.memory)

    def display_graph(self):
        return Image(self.graph.get_graph().draw_mermaid_png())

    def question_answer_node(
        self, state: AgentState
    ) -> Command[Literal["task_decoder", "the_end"]]:
        logger.info("Question Answer Node")
        development_concurrent_time = state.get("development_concurrent_time", 0)
        message = QUESTION_ANSWERING_INSTRUCTION.format(
            development_concurrent_time=development_concurrent_time,
            task=state.get("task", ""),
            plan=state.get("plan", "Day:1"),
            suggestion=state.get("messages", ["No suggestions returned yet"])[-1],
        )
        conversation_history = state["conversation_history"] or []
        human_msg = HumanMessage(content=state["message"])
        message = conversation_history + [
            SystemMessage(content=message),
            human_msg,
        ]
        msg_ext = []
        ai_msg_ok = False
        for __ in range(3):
            msg_attempt = message.copy() + msg_ext
            response = self.llm.invoke(msg_attempt)
            try:
                py_response = dict(ast.literal_eval(response.content))
                development = py_response["development"]
                ai_message = py_response["message"]
                assert isinstance(development, bool) and isinstance(
                    ai_message, str
                ), "The 'development' key should have a True or False value, and the 'message' key should be a valid string value."
                ai_msg_ok = True
            except ValueError as e:
                msg_ext = [
                    SystemMessage(
                        content="Wrong formated response.The response returned is {}. Error provided: {}".format(
                            response.content, e
                        )
                    )
                ]
            except KeyError as e:
                msg_ext = [
                    SystemMessage(
                        content="Wrong formated response, {}. The response returned is {}".format(
                            e, response.content
                        )
                    )
                ]
            except AssertionError as e:
                msg_ext = [
                    SystemMessage(
                        content="Wrong formated response, {}. The response returned is {}".format(
                            e, response.content
                        )
                    )
                ]

        assert (
            ai_msg_ok
        ), "The AI message was not generated correctly for 'question_answer_node' requirements."

        # Replacement for a conditional edge function
        if development:
            goto = "task_decoder"
            add_to_history = [human_msg]
        else:
            goto = "the_end"
            add_to_history = [human_msg, AIMessage(content=ai_message)]

        return Command(
            # this is the state update
            update={
                "conversation_history": add_to_history,
                "development_concurrent_time": development_concurrent_time + 1,
            },
            # this is a replacement for an edge
            goto=goto,
        )

    def task_decoder_node(self, state: AgentState):
        logger.info("Task Decoder Node")
        message = TASK_DECODER_INSTRUCTION.format(
            task=state.get("task", state["message"]),
            plan=state.get("plan", "Day:1"),
        )
        message = state["conversation_history"] + [
            SystemMessage(content=message),
        ]
        response = self.llm.invoke(message)
        return {
            "task": response.content,
        }

    def researcher_node(self, state: AgentState):
        logger.info("Researcher Node")
        message = RESEARCH_INSTRUCTION.format(
            task=state["task"],
            plan=state.get("plan", "Day:1"),
        )
        message = state["conversation_history"] + [
            SystemMessage(content=message),
        ]
        response = self.llm_tools.invoke(message)
        return {
            "messages": [response],
        }

    def tool_parser_node(self, state: AgentState):
        logger.info("Tool Parser Node")
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        docs_content = "\n\n".join(doc.content for doc in recent_tool_messages[::-1])
        return {"information": [AnyMessage(content=docs_content)]}
