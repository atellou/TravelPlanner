from typing import List, Type, Optional
import re
import ast
import time

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_core.tools import BaseTool
from langchain_core.documents import Document

from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


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
        logger.info("Ingesting {} documents in vectore store...".format(len(documents)))
        self.vector_store.add_documents(documents=documents)

    def _run(
        self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        logger.warning(f"Searching information for {question}")
        retrieved_docs = self.vector_store.similarity_search(question, k=2)
        serialized = [
            {f"Source: {doc.metadata}", f"Content: {doc.page_content}"}
            for doc in retrieved_docs
        ]
        return serialized
