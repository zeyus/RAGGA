import re
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
from langchain.schema.runnable import RunnableParallel

from ragga.pipeline.config import Config, Configurable
from ragga.pipeline.documents import VectorDatabase
from ragga.pipeline.encoder import Encoder


class Generator(Configurable):
    """Generator, aka LLM, to provide an answer based on some question and context"""

    _config_key = "generator"
    _default_config = MappingProxyType(
        {
            "llm_path": "llm",
            "context_length": 1024,
            "temperature": 0.9,
            "gpu_layers": 2,
            "max_tokens": 128,
            "n_batch": 256,
            "compress": False,
        }
    )

    def __init__(
        self, conf: Config, template: PromptTemplate, vectorstore: VectorDatabase, encoder: Encoder
    ) -> None:
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
        self._merge_default_kwargs(default_kwargs, "model_kwargs")
        # load Llama from local file
        self.llm = LlamaCpp(
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
        self.max_prompt_length = (
            self.config["generator"]["context_length"] - self.config["generator"]["max_tokens"] - 100 - 1
        )  # placeholder, will need to calculate this
        self.prompt = template
        self.output_parser = StrOutputParser()
        self.vectorstore = vectorstore
        self.retriever = vectorstore.db.as_retriever(search_type="similarity", k=self.config["retriever"]["num_docs"])
        self.encoder = encoder
        self.embeddings_filter = EmbeddingsFilter(embeddings=encoder.encoder, similarity_threshold=0.76)
        self.reorder = LongContextReorder()
        self.pipeline = DocumentCompressorPipeline(transformers=[self.embeddings_filter, self.reorder])
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.pipeline, base_retriever=self.retriever
        )
        self.inputs = RunnableParallel(
            {
                "question": itemgetter("question"),
                "context": itemgetter("question")
                | (self.compression_retriever if self.config[self._config_key]["compress"] else self.retriever)
                | self.condense_context,
            }
        )
        # RunnablePassthrough(func=lambda x: print(x))
        self.chain = self.inputs | self.prompt | self.llm | self.output_parser
        self.context_split = re.compile(r"^--------------------$", re.MULTILINE)

    def condense_context(self, ctx: list[Document]) -> str:
        """
        Condense the context into a single string
        Args:
            ctx (list[Document]): list of documents
        Returns:
            str: condensed context
        """
        condensed = "\n".join([d.page_content for d in ctx])
        if len(condensed) > self.max_prompt_length:
            condensed = condensed[: self.max_prompt_length]
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

        return self.chain.invoke({"question": question})
