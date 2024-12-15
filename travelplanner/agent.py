from typing import TypedDict, List

from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage

from pydantic import BaseModel
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
    plan_information: str
    revision_number: int
    max_revisions: int
    plan: str
    critique: str


class AgentPlanner:
    def __init__(
        self, model_name="gemini-1.5-flash", embedding_model="text-embedding-004"
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        llm = ChatVertexAI(
            model=model_name,
        )
        embeddings = VertexAIEmbeddings(model=embedding_model)
        vector_store = InMemoryVectorStore(embeddings)

        self.llm = llm
        self.vector_store = vector_store
        self.retriever = self.vector_store.as_retriever()
        self.tools = ToolNode([self.retrieve])
        self.llm_tools = self.llm.bind_tools([self.retrieve])
        self.define_graph()

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
        self.memory = SqliteSaver.from_conn_string(":memory:")
        self.graph = builder.compile(checkpointer=self.memory)

    def load_reference_information(self, chunks):
        """Load reference information into the vector store."""
        self.vector_store.add_documents(documents=chunks)

    def display_graph(self):
        Image(self.graph.get_graph().draw_png())

    def question_node(self, state: AgentState):
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
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is some possible usefull information:\n\n{state['plan_information']}"
        )
        messages = [
            SystemMessage(
                content=REACT_REFLECT_PLANNER_INSTRUCTION.format(
                    reflections=state.get("critique", ""),
                    content=state.get("plan_information", ""),
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
                content=f"{state['plan']}\n\nHere is some possible usefull information:\n\n{state['plan_information']}"
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
