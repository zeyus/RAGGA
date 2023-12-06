from types import MappingProxyType

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from ragga.pipeline.config import Config, Configurable


class VectorDatabase(Configurable):
    """FAISS based Vector database"""

    _config_key = "retriever"
    _default_config = MappingProxyType({
        "splitting": {
            "chunk_size": 256,
            "chunk_overlap": 16,
        },
        "num_docs": 4,
    })
    db: FAISS

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        self.retriever = FAISS
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.config[self._config_key]["splitting"]["chunk_size"],
            chunk_overlap=self.config[self._config_key]["splitting"]["chunk_overlap"],
        )

    def create_passages_from_documents(self, documents: list) -> list:
        """
        Splits the documents into passages of a certain length
        Args:
            documents (list): list of documents
        Returns:
            list: list of passages
        """
        return self.text_splitter.split_documents(documents)

    def store_passages_db(self, passages: list, encoder: Embeddings) -> None:
        """
        Store passages in vector database in embedding format
        Args:
            passages (list): list of passages
            encoder (Embeddings): encoder to convert passages into embeddings
        """
        self.db = self.retriever.from_documents(passages, encoder)

    def retrieve_most_similar_document(self, question: str, k: int) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to query
        Returns:
            str: most similar document
        """
        docs: list[Document] = self.db.similarity_search(question, k=k)
        strdocs: list[str] = [d.page_content for d in docs]

        return "\n".join(strdocs)
