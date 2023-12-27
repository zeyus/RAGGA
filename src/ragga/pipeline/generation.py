import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from langchain.callbacks.manager import CallbackManager
from langchain.docstore.document import Document
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.llms.llamacpp import LlamaCpp
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from ragga.core.config import Config, Configurable
from ragga.core.dispatch import PropertyWrapper
from ragga.crafting.commands import extract_keywords
from ragga.crafting.prompt import Prompt
from ragga.pipeline.documents import VectorDatabase

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class ModelResponseHandler(BaseCallbackHandler):
    """Callback handler for model output."""


    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model_response = PropertyWrapper[str](initial_value=None)

    @property
    def model_response(self) -> str | None:
        """Get the model response."""
        return self._model_response.__get__(None, None)

    @model_response.setter
    def model_response(self, value: str | None) -> None:
        """Set the model response."""
        self._model_response.__set__(self, value)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Run when LLM starts running."""
        logging.info("LLM started...")

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list["BaseMessage"]],  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Run when LLM starts running."""
        logging.info("Chat model started...")

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:  # noqa: ARG002
        """Run on new LLM token. Only available when streaming is enabled."""
        self.model_response = token  # type: ignore

    def on_llm_end(self, response: "LLMResult", **kwargs: Any) -> None:  # noqa: ARG002
        """Run when LLM ends running."""
        logging.info("LLM ended...")

    def on_llm_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Run when chain starts running."""
        logging.info("Chain started...")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:  # noqa: ARG002
        """Run when chain ends running."""
        logging.info("Chain ended...")

    def on_chain_error(self, error: "BaseException", **kwargs: Any) -> None:  # noqa: ARG002
        """Run when chain errors."""
        logging.error(f"Chain error: {error}")

    def on_tool_start(
        self, serialized: dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: "AgentAction", **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: "AgentFinish", **kwargs: Any) -> None:
        """Run on agent end."""


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
            "compress": True,
            "similarity_threshold": 0.8,
        }
    )

    _retriever: ContextualCompressionRetriever | VectorStoreRetriever

    def __init__(
        self, conf: Config, prompt: Prompt, vectorstore: VectorDatabase) -> None:
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
        self._keywords = {
            "command": "",
            "where": "",
            "what": "",
            "known": False,
        }
        self._last_query = None
        self._model_response_handler = ModelResponseHandler()
        self._merge_default_kwargs(default_kwargs, "model_kwargs")
        self._merge_default_kwargs(search_kwargs, "search_kwargs")
        # load Llama from local file
        self._llm = LlamaCpp(
            model_path=self.config[self._config_key]["llm_path"],
            n_ctx=self.config[self._config_key]["context_length"],
            temperature=self.config[self._config_key]["temperature"],
            n_gpu_layers=self.config[self._config_key]["gpu_layers"],
            max_tokens=self.config[self._config_key]["max_tokens"],
            callbacks=CallbackManager([self._model_response_handler]),
            n_batch=self.config[self._config_key]["max_tokens"],
            **self.config[self._config_key]["model_kwargs"],
        )
        # create prompt template
        self._max_prompt_length = (
            self.config[self._config_key]["context_length"] - self.config[self._config_key]["max_tokens"] - 100 - 1
        )  # placeholder, will need to calculate this
        self._prompt = prompt
        self._template = prompt.get_prompt()
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
        self._inputs = RunnableParallel(  # type: ignore
            {
                "user": lambda _x: self._prompt.user_name,
                "question": itemgetter("question"),
                "date": lambda _x: datetime.now(UTC).strftime("%Y-%m-%d"),
                "context": itemgetter("question")
                | RunnableLambda(self._prepare_query)
                | self._retriever
                | self.condense_context,
            }  # type: ignore
        )
        self._llm_with_stop = self._llm.bind(stop=[f"\n{self._prompt.user_name}:"])
        # RunnablePassthrough(func=lambda x: print(x))
        self._chain = self._inputs | self._template | self._llm_with_stop | self._output_parser

    def _prepare_query(self, x: str) -> str:
        """
        Prepare the query for the retriever, limiting to keywords that are hopefully relevant.
        """
        # Get relevant keywords
        if self._keywords["what"] is not None:
            if self._keywords["known"] or self._keywords["where"]  is None:
                logging.info(f"Using keyword {self._keywords['what'] } as query")
                return self._keywords["what"]
            logging.info(f"Using keywords {self._keywords['what'] } and {self._keywords['where']} as query")
            return f"{self._keywords['what'] } {self._keywords['where']}"
        logging.info(f"Using full query {x}")
        return x

    def condense_context(self, ctx: list[Document]) -> str:
        """
        Condense the context into a single string
        Args:
            ctx (list[Document]): list of documents
        Returns:
            str: condensed context
        """
        condensed = "\n".join([d.page_content for d in ctx])
        if len(condensed) > self._max_prompt_length:
            logging.warning(f"Context too long, truncating to {self._max_prompt_length} characters")
            condensed = condensed[: self._max_prompt_length]
        return condensed

    def _query_keywords(self, question: str) -> str:
        """
        Query the keywords from the question
        Args:
            question (str): user's question
        """
        known_locations = set({"notes", "note", "writing", "writings"})
        kw = extract_keywords(question, known_locations)
        if kw[0] in ["redo", "repeat", "again"] and self._last_query is not None:
            kw = extract_keywords(self._last_query, known_locations)
            question = self._last_query
        else:
            self._last_query = question
        self._keywords["command"] = kw[0]
        self._keywords["where"] = kw[1]
        self._keywords["what"] = kw[2]
        self._keywords["known"] = kw[3]
        return question


    def get_answer(self, question: str) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            context (str): most similar document retrieved
            question (str): user's question
        Returns:
            Iterator[str]: llm answer
        """
        question = self._query_keywords(question)
        return self._chain.invoke({"question": question})

    def get_answer_stream(self, question: str) -> Iterator[str]:
        """
        Get the answer from llm based on context and user's question
        Args:
            context (str): most similar document retrieved
            question (str): user's question
        Returns:
            Iterator[str]: llm answer
        """
        question = self._query_keywords(question)
        return self._chain.stream({"question": question})

    def response_handler(self) -> ModelResponseHandler:
        """Get the response handler"""
        return self._model_response_handler

    def subscribe(self, subscriber: Callable[[str], None]):
        """Subscribe to updates of the associated property."""
        self._model_response_handler._model_response.subscribe(subscriber)
