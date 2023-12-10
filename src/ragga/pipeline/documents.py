import typing as t
from pathlib import Path
from types import MappingProxyType

from langchain.docstore.document import Document
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from ragga.core.config import Config, Configurable
from ragga.pipeline.encoder import Encoder


class VectorDatabase(Configurable):
    """FAISS based Vector database"""

    _config_key = "retriever"
    _default_config = MappingProxyType({
        "splitting": {
            "chunk_size": 256,
            "chunk_overlap": 16,
        },
        "num_docs": 4,
        "cache_dir": ".cache",
    })
    _db: FAISS | None = None
    _vector_store: type[FAISS]
    _retriever: VectorStoreRetriever | None = None
    _passages: list[Document] | None = None
    _documents: list[Document]
    _db_is_stale: bool = True
    _encoder: Encoder
    _record_manager: SQLRecordManager


    def __init__(self, conf: Config, encoder: Encoder, namespace: str | None = None) -> None:
        super().__init__(conf)
        self._text_splitter = CharacterTextSplitter(
            chunk_size=self.config[self._config_key]["splitting"]["chunk_size"],
            chunk_overlap=self.config[self._config_key]["splitting"]["chunk_overlap"],
        )
        self._index_db_path = self.config[self.key]["cache_dir"] + "/.record_index.db"
        self._vector_db_path = Path(self.config[self.key]["cache_dir"] + "/.faiss")
        self._encoder = encoder
        self._vector_store = FAISS
        self._namespace = namespace if namespace is not None else self._config_key
        self._record_manager = SQLRecordManager(
            namespace=self._namespace,
            db_url=f"sqlite:///{self._index_db_path}",
        )

        if not self._vector_db_path.exists():
            self._vector_db_path.mkdir(parents=True)

    @property
    def documents(self) -> list[Document]:
        return self._documents

    @documents.setter
    def documents(self, documents: list[Document]) -> None:
        """
        Adds documents to the database
        Args:
            documents (list): list of documents
        """
        self._documents = documents
        self._db_is_stale = True

    @property
    def passages(self) -> list[Document]:
        if self._passages is None or self._db_is_stale:
            self._passages = self._text_splitter.split_documents(self.documents)
        return self._passages

    @property
    def db(self) -> VectorStore:
        if self._db is None or self._db_is_stale:
            self._create_db()
        if self._db is None:
            msg = "Database failed to initialize."
            raise ValueError(msg)
        return self._db

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def _add_to_index(self, documents: list[Document], mode: t.Literal["incremental", "full"] = "incremental") -> None:
        """
        Adds documents to the index
        Args:
            documents (list): list of documents
        """
        if self._db is not None:
            index(
                documents,
                record_manager=self._record_manager,
                vector_store=self._db,
                cleanup=mode,
                source_id_key="source",
            )

    def _clear_index(self) -> None:
        """
        Clears the index
        """
        self._add_to_index([], mode="full")

    def _update_index(self) -> None:
        if self._db is None:
            msg = "Database failed to initialize."
            raise ValueError(msg)
        self._add_to_index(self.passages)
        self._db.save_local(folder_path=str(self._vector_db_path), index_name="index")
        self._db_is_stale = False

    def _create_db(self) -> None:
        """
        Store passages in vector database in embedding format
        """
        self._record_manager.create_schema()
        if self._db is None:
            if (self._vector_db_path / "index.faiss").exists():
                self._db = self._vector_store.load_local(
                    folder_path=str(self._vector_db_path),
                    embeddings=self._encoder.embeddings,
                    index_name="index",
                )
            else:
                self._db = self._vector_store.from_texts(["ZEROINDEX"], self._encoder.embeddings)
        self._update_index()

    def merge_documents(self, documents: list[Document]) -> None:
        """
        Merges documents with existing documents
        Args:
            documents (list): list of documents
        """
        self._documents.extend(documents)
        self._db_is_stale = True
        self._update_index()

    def get_similar_docs(self, question: str, k: int | None = None) -> str:
        """
        Retrieves the most similar document for a certain question
        Args:
            question (str): user question
            k (int): number of documents to return
                     defaults to config value if None
        Returns:
            str: most similar document
        """
        k = self.config[self._config_key]["num_docs"] if k is None else k
        docs: list[Document] = self.db.similarity_search(question, k=k)
        strdocs: list[str] = [d.page_content for d in docs]

        return "\n".join(strdocs)

    @property
    def retriever(self) -> VectorStoreRetriever:
        if self._retriever is None:
            self._retriever = self.db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config[self._config_key]["num_docs"]},
            )
        return self._retriever
