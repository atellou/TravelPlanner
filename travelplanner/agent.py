import os
import getpass
from typing import TypedDict, Annotated, List
import langchain

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from travelplanner.prompt import (
    QUESTION_PLANNER_INSTRUCTION,
    RESEARCHER_PLAN_INSTRUCTION,
    REACT_REFLECT_PLANNER_INSTRUCTION,
    REFLECT_INSTRUCTION,
    RESEARCH_CRITIQUE_PROMPT,
)

from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ChatMessage,
)
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    task: str
    travel_questions: str
    plan_information: str
    draft: str
    revision_number: int
    max_revisions: int
    plan: str
    critique: str
    content: List[str]


class AgentPlanner:
    def __init__(
        self, model_name="gemini-1.5-flash", embedding_model="text-embedding-004"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        llm = ChatVertexAI(model=model_name)
        embeddings = VertexAIEmbeddings(model=embedding_model)
        vector_store = InMemoryVectorStore(embeddings)

        self.llm = llm
        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever()

        self.llm_tools = self.llm.bind_tools([self.retrieve])

        self.reflect_prompt = langchain.prompts.PromptTemplate(
            input_variables=["text", "query", "scratchpad"],
            template=REFLECT_INSTRUCTION,
        )

        self.agent_prompt = langchain.prompts.PromptTemplate(
            input_variables=["text", "query", "reflections", "scratchpad"],
            template=REACT_REFLECT_PLANNER_INSTRUCTION,
        )

        self.graph_builder = StateGraph(MessagesState)

        tools = ToolNode([self.retrieve])

    def question_planner_node(self, state: AgentState):
        messages = [
            SystemMessage(content=QUESTION_PLANNER_INSTRUCTION),
            HumanMessage(content=state["task"]),
        ]
        response = self.llm.with_structured_output(Queries).invoke(messages)

        return {"travel_questions": response.content}

    def researcher_planner_node(self, state: AgentState):

        response = self.llm_tools.invoke(
            [
                SystemMessage(content=RESEARCHER_PLAN_INSTRUCTION),
                HumanMessage(
                    content=f"{state['task']}\n\nThese are some travel questions to"
                    + f" take into account:\n\n{state['travel_questions'].queries}"
                ),
            ]
        )
        return {"plan_information": response.content}

    def generation_node(self, state: AgentState):
        content = "\n\n".join(state["content"] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is some possible usefull information:\n\n{state['plan_information']}"
        )
        messages = [
            SystemMessage(
                content=REACT_REFLECT_PLANNER_INSTRUCTION.format(content=content)
            ),
            user_message,
        ]
        response = self.llm.invoke(messages)
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECT_INSTRUCTION),
            HumanMessage(
                content=f"{state['draft']}\n\nHere is some possible usefull information:\n\n{state['plan_information']}"
            ),
        ]
        response = self.llm.invoke(messages)
        return {"critique": response.content}

    def research_critique_node(self, state: AgentState):

        response = self.llm_tools.invoke(
            [
                SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
                HumanMessage(
                    content=f"{state['task']}"
                    + f"n\nInformation obtained before:\n\n{state['plan_information']}"
                    + f"\n\n Critique provided by expert:\n\n{state['critique']}"
                ),
            ]
        )

        content = state["plan_information"] or []
        content += f"Complementary information:{response.content}"
        return {"plan_information": content}

    def should_continue(state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

    @tool(response_format="content_and_artifact")
    def retrieve(self, query: str):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    def query_or_respond(self, state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = self.llm.bind_tools([self.retrieve])
        response = llm_with_tools.invoke(state["messages"])
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}

    def generate(self, state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}

    def run(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return docs
