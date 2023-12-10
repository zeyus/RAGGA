from datetime import UTC, datetime
from operator import itemgetter
from types import MappingProxyType

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

from ragga.core.config import Config, Configurable
from ragga.pipeline.documents import VectorDatabase


class Generator(Configurable):
    """Generator, aka LLM, to provide an answer based on some question and context"""

    _config_key = "generator"
    _default_config = MappingProxyType(
        {
            "llm_path": "llm",
            "context_length": 1024,
            "temperature": 0.9,
            "gpu_layers": 50,
            "max_tokens": 128,
            "n_batch": 256,
            "compress": False,
            "similarity_threshold": None,
        }
    )

    def __init__(
        self, conf: Config, template: PromptTemplate, vectorstore: VectorDatabase) -> None:
        super().__init__(conf)

        default_kwargs = {
            "streaming": True,
            "verbose": True,
            "f16_kv": True,
            "echo": False,
            "use_mlock": False,
            "client": None,
            "n_parts": -1,
            "n_threads": None,
            "logprobs": None,
            "seed": -1,
            "logits_all": False,
            "vocab_only": False,
            "suffix": None,
        }
        search_kwargs = {
            "k": self.config[vectorstore.key]["num_docs"],
        }
        self._merge_default_kwargs(default_kwargs, "model_kwargs")
        self._merge_default_kwargs(search_kwargs, "search_kwargs")
        # load Llama from local file
        self._llm = LlamaCpp(
            model_path=self.config[self._config_key]["llm_path"],
            n_ctx=self.config[self._config_key]["context_length"],
            temperature=self.config[self._config_key]["temperature"],
            n_gpu_layers=self.config[self._config_key]["gpu_layers"],
            max_tokens=self.config[self._config_key]["max_tokens"],
            callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
            n_batch=self.config[self._config_key]["max_tokens"],
            **self.config[self._config_key]["model_kwargs"],
        )
        # create prompt template
        self._max_prompt_length = (
            self.config[self._config_key]["context_length"] - self.config[self._config_key]["max_tokens"] - 100 - 1
        )  # placeholder, will need to calculate this
        self._prompt = template
        self._output_parser = StrOutputParser()
        self._vectorstore = vectorstore
        self._embeddings_filter = EmbeddingsFilter(
            embeddings=self._vectorstore.encoder.embeddings,
            similarity_threshold=self.config[self._config_key]["similarity_threshold"])
        self._reorder = LongContextReorder()
        self._pipeline = DocumentCompressorPipeline(transformers=[self._embeddings_filter, self._reorder])
        if self.config[self._config_key]["compress"]:
            self._compression_retriever = ContextualCompressionRetriever(
                base_compressor=self._pipeline, base_retriever=self._vectorstore.retriever
            )
            self._retriever = self._compression_retriever
        else:
            self._retriever = self._vectorstore.retriever
        self._inputs = RunnableParallel(
            {
                "question": itemgetter("question"),
                "date": lambda x: datetime.now(UTC).strftime("%Y-%m-%d"),  # noqa: ARG005
                "context": itemgetter("question")
                | self._retriever
                | self.condense_context,
            }
        )
        # RunnablePassthrough(func=lambda x: print(x))
        self._chain = self._inputs | self._prompt | self._llm | self._output_parser

    def condense_context(self, ctx: list[Document]) -> str:
        """
        Condense the context into a single string
        Args:
            ctx (list[Document]): list of documents
        Returns:
            str: condensed context
        """
        condensed = "\n".join([d.page_content for d in ctx])
        print(ctx[0].metadata)
        print(f"Condensed context: {condensed}, length: {len(condensed)}")
        if len(condensed) > self._max_prompt_length:
            print(f"Context too long, truncating to {self._max_prompt_length} characters")
            condensed = condensed[: self._max_prompt_length]
        return condensed

    def get_answer(self, question: str) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            context (str): most similar document retrieved
            question (str): user's question
        Returns:
            Iterator[str]: llm answer
        """

        return self._chain.invoke({"question": question})
