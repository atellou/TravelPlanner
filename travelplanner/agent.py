from typing import TypedDict, List, Type, Optional
import sqlite3
import re
import ast

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document

from pydantic import BaseModel, Field
from IPython.display import Image, display

from prompt import (
    QUESTION_PLANNER_INSTRUCTION,
    RESEARCHER_PLAN_INSTRUCTION,
    REACT_REFLECT_PLANNER_INSTRUCTION,
    REFLECT_INSTRUCTION,
    RESEARCH_CRITIQUE_PROMPT,
)


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    task: str
    travel_questions: str
    content: str
    revision_number: int
    max_revisions: int
    plan: str
    critique: str


class VectorStoreQuestion(BaseModel):
    question: str = Field(
        description="Question provided as reference to search information"
    )


class VectorRetrieverTool(BaseTool):
    name: str = "KbdRetreiver"
    description: str = "Used to retreive information from a knowledge base"
    args_schema: Type[BaseModel] = VectorStoreQuestion
    return_direct: bool = True
    embeddings: VertexAIEmbeddings = "text-embedding-004"
    vector_store: InMemoryVectorStore = None

    def __init__(
        self,
        data: List[dict],
        description: str,
        embedding_model: str = "text-embedding-004",
    ):
        super().__init__()
        self.embeddings = VertexAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self._load_reference_information(data)
        if description is not None:
            self._add_description(description)

    def _add_description(self, description: str):
        """Add description to the vector store."""
        msg = "\n\nThe data contained corresponds to:\n\n"
        self.description += msg + description.capitalize()

    def _load_reference_information(self, data):
        """Load reference information into the vector store."""
        documents = []
        for information in data:
            # Drop duplicated spaces
            information = re.sub(r"[ ]{2,}", " ", information)
            # Get objects by dict
            information = re.findall(r"\{.*?\}", information)
            # Format to json quotes
            documents.extend(
                [
                    Document(
                        page_content=item["Content"],
                        metadata={"description": item["Description"]},
                    )
                    for item in map(ast.literal_eval, information)
                ]
            )
        self.vector_store.add_documents(documents=documents)

    def _run(
        self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        retrieved_docs = self.vector_store.similarity_search(question, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # @tool(response_format="content_and_artifact")
    # def retrieve(query: str):
    #     """Retrieve information related to a query."""
    #     retrieved_docs = self.vector_store.similarity_search(query, k=2)
    #     serialized = "\n\n".join(
    #         (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
    #         for doc in retrieved_docs
    #     )
    #     return serialized, retrieved_docs


class AgentPlanner:
    def __init__(self, tools: List[BaseTool], model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.llm = ChatVertexAI(
            model=model_name,
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
        builder = self.graph_builder = StateGraph(AgentState)
        builder.add_node("question", self.question_node)
        builder.add_node("research", self.researcher_planner_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        #
        builder.set_entry_point("question")
        # Edges
        builder.add_edge("question", "research")
        builder.add_edge("research", "generate")
        builder.add_conditional_edges(
            "generate", self.should_continue, {END: END, "reflect": "reflect"}
        )
        builder.add_edge("reflect", "research_critique")
        builder.add_edge("research_critique", "generate")

        self.conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        return builder.compile(checkpointer=self.memory)

    def display_graph(self):
        return Image(self.graph.get_graph().draw_png())

    def question_node(self, state: AgentState):
        messages = [
            SystemMessage(content=QUESTION_PLANNER_INSTRUCTION),
            HumanMessage(content=state["task"]),
        ]
        response = self.llm.with_structured_output(Queries).invoke(messages)

        return {"travel_questions": response.queries}

    def researcher_planner_node(self, state: AgentState):

        response = self.llm_tools.invoke(
            [
                SystemMessage(content=RESEARCHER_PLAN_INSTRUCTION),
                HumanMessage(
                    content=f"{state['task']}\n\nThese are some travel questions to"
                    + f" take into account:\n\n{state['travel_questions']}\n\n"
                ),
            ]
        )
        content = state["content"] or "Complementary information:"
        content += f"\n\n{response.content}"
        return {"content": content}

    def generation_node(self, state: AgentState):
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is some possible usefull information:\n\n{state['content']}"
        )
        messages = [
            SystemMessage(
                content=REACT_REFLECT_PLANNER_INSTRUCTION.format(
                    reflections=state.get("critique", ""),
                    content=state.get("content", ""),
                )
            ),
            user_message,
        ]
        response = self.llm.invoke(messages)
        return {
            "plan": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
        }

    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=REFLECT_INSTRUCTION),
            HumanMessage(
                content=f"{state['plan']}\n\nHere is some possible usefull information:\n\n{state['content']}"
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
                    + f"n\nInformation obtained before:\n\n{state['content']}"
                    + f"\n\n Critique provided by expert:\n\n{state['critique']}"
                ),
            ]
        )

        content = state["content"] or "Complementary information:"
        content += f"\n\n{response.content}"
        return {"content": content}

    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"
