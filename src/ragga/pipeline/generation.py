import logging
from collections.abc import Callable, Iterator
from io import StringIO
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from langchain.callbacks.manager import CallbackManager
from langchain.docstore.document import Document
from langchain.llms.llamacpp import LlamaCpp

# from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.base import Runnable
from langchain_experimental.chat_models import Llama2Chat

# from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore
from ragga.core.config import Config, Configurable
from ragga.core.dispatch import PropertyWrapper
from ragga.core.io import store_stdout_stderr
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
        logging.debug("LLM started...")

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
        logging.debug("LLM ended...")

    def on_llm_error(self, error: "BaseException", **kwargs: Any) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any  # noqa: ARG002
    ) -> None:
        """Run when chain starts running."""
        logging.debug("Chain started...")

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:  # noqa: ARG002
        """Run when chain ends running."""
        logging.debug("Chain ended...")

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
            "hf_tokenizer": "",
            "context_length": 1024,
            "temperature": 0.9,
            "gpu_layers": 50,
            "max_tokens": 128,
            "n_batch": 256,
            "compress": True,
            "similarity_threshold": 0.8,
            "llama": False,
            "autoflush": True,
        }
    )

    _retriever: BaseRetriever

    def __init__(
        self, conf: Config,
        prompt: Prompt,
        vectorstore: VectorDatabase,
        websearch: BaseRetriever | None = None,
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
        search_kwargs = {
            "k": self.config[vectorstore.key]["num_docs"],
        }
        self._keywords: dict[str, str | bool | None] = {
            "command": "",
            "where": "",
            "what": "",
            "known": False,
        }
        self._model_stdout: StringIO = StringIO()
        self._model_stderr: StringIO = StringIO()
        self._web_retriever: BaseRetriever | None = websearch
        self._known_locations: set[str] = set({"notes", "note", "writing", "writings"})
        self._last_query: str | None = None
        self._question: str | None = None
        self._last_context: str | None = None
        self._last_output: str = ""
        self._web_search: bool = False
        self._model_response_handler = ModelResponseHandler()
        self._merge_default_kwargs(default_kwargs, "model_kwargs")
        self._merge_default_kwargs(search_kwargs, "search_kwargs")
        self._autoflush: bool = True if self.config[self._config_key]["autoflush"] else False
        if not self._autoflush:
            logging.debug(
                "Autoflush disabled, please remember to flush stdout and stderr "
                "buffers maunally with {__class__}.flush_stdout_stderr()"
            )
        with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
            # load Llama from local file
            self._llm: LlamaCpp = LlamaCpp(
                model_path=self.config[self._config_key]["llm_path"],
                n_ctx=self.config[self._config_key]["context_length"],
                temperature=self.config[self._config_key]["temperature"],
                n_gpu_layers=self.config[self._config_key]["gpu_layers"],
                max_tokens=self.config[self._config_key]["max_tokens"],
                callbacks=CallbackManager([self._model_response_handler]),
                n_batch=self.config[self._config_key]["max_tokens"],
                **self.config[self._config_key]["model_kwargs"],
            )
        self._llm_with_wrap:  Runnable[LanguageModelInput, str | BaseMessage]
        # self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        #     self.config[self._config_key]["hf_tokenizer"]
        # )
        if self.config[self._config_key]["llama"]:
            self._llm_with_wrap = Llama2Chat(
                llm=self._llm,
            )
        else:
            self._llm_with_wrap = self._llm
        # create prompt template
        self._max_prompt_length: int = (
            self.config[self._config_key]["context_length"] - self.config[self._config_key]["max_tokens"] - 100 - 1
        )  # placeholder, will need to calculate this
        self._prompt: Prompt = prompt
        self._template: BasePromptTemplate | None = None
        self._output_parser: BaseTransformOutputParser = StrOutputParser()
        self._vectorstore: VectorDatabase = vectorstore
        self._embeddings_filter = EmbeddingsFilter(
            embeddings=self._vectorstore.encoder.embeddings,
            similarity_threshold=self.config[self._config_key]["similarity_threshold"])
        self._reorder = LongContextReorder()
        self._pipeline = DocumentCompressorPipeline(transformers=[self._embeddings_filter, self._reorder])
        self._retriever: BaseRetriever
        self._docs: list[Document] | None = None
        # This could be instead replaced with a different compression method (without LLM).
        # if self.config[self._config_key]["compress"]:
        #     self._compression_retriever = ContextualCompressionRetriever(
        #         base_compressor=self._pipeline, base_retriever=self._vectorstore.retriever
        #     )
        #     self._retriever = self._compression_retriever
        # else:
        self._retriever = self._vectorstore.retriever
        self._inputs: Runnable = RunnableParallel(  # type: ignore
            {
                "question": itemgetter("question"),
                "context": itemgetter("question")
                | RunnableLambda(func=self._prepare_query)
                | RunnablePassthrough(func=self._set_template)
                | RunnableBranch(
                    (lambda x: self._use_web_retriever(x), self._web_retriever),  # type: ignore
                    self._retriever
                )
                | self.condense_context,
            }  # type: ignore
        )

        self._gen_chain: Runnable = (
            self._inputs
            | RunnableLambda(func=self._get_template)
            | RunnablePassthrough(func=self._store_context)
            | self._llm_with_wrap
            | self._output_parser
            | RunnablePassthrough(func=self._store_output)
        )
        self._continue_chain: Runnable = (
            RunnableLambda(func=self._create_continuation)
            | RunnablePassthrough(func=self._store_context)
            | self._llm_with_wrap
            | self._output_parser
            | RunnablePassthrough(func=self._store_output)
        )

        self._chain: Runnable = self._gen_chain

    def _create_continuation(self, _x: Any) -> str:
        """Create a prompt based on the last context and output"""
        if self._last_context is None:
            msg = "No context to continue"
            raise ValueError(msg)
        prompt: str = self._last_context + self._last_output
        logging.debug("Calculating continuation length...")
        enc_prompt: list[int] = self._llm.client.tokenize(prompt.encode())
        if len(enc_prompt) > self._max_prompt_length:
            logging.debug("Continuation too long, truncating start...")
            bprompt: bytes = self._llm.client.detokenize(enc_prompt[-self._max_prompt_length:])
            prompt = bprompt.decode()
        logging.debug("Continuation length calculated")
        return prompt

    def _get_template(self, _x: Any) -> BasePromptTemplate:
        """Get the template"""
        if self._template is None:
            msg = "No template set."
            raise ValueError(msg)
        return self._template

    def _set_template(self, _x: str) -> None:
        """Get the template"""
        if self._use_web_retriever():
            self._template = self._prompt.get_prompt(web=True)
        if isinstance(self._last_context, str) and len(self._last_context.strip()) > 0:
            self._template = self._prompt.get_prompt(docs=True)
        self._template = self._prompt.get_prompt()

    def _can_continue(self) -> bool:
        """Check if the context can be continued"""
        if self._last_context is None:
            return False
        return True

    def _store_context(self, ctx: ChatPromptValue | str) -> None:
        """
        Store the context
        Args:
            ctx (str): context
        """
        if isinstance(ctx, ChatPromptValue):
            ctx = ctx.to_string()
        self._last_context = ctx
        self._last_output = ""

    def _store_output(self, output: str) -> None:
        """
        Store the output
        Args:
            output (str): output
        """
        self._last_output += output

    def _use_web_retriever(self, _: Any = None) -> bool:
        """Get the retriever dropin"""
        if self._web_search and self._web_retriever is not None:
            logging.info("Using web retriever")
            return True
        logging.info("Using vector retriever")
        return False

    def _prepare_query(self, x: str) -> str:
        """
        Prepare the query for the retriever, limiting to keywords that are hopefully relevant.
        """
        # Get relevant keywords
        if self._keywords["what"] is not None:
            if self._keywords["known"] or self._keywords["where"]  is None:
                logging.info(f"Using keyword {self._keywords['what'] } as query")
                return self._keywords["what"]  # type: ignore
            logging.info(f"Using keywords {self._keywords['what'] } and {self._keywords['where']} as query")
            return f"{self._keywords['what'] } {self._keywords['where']}"
        logging.info(f"Using full query {x}")
        return x

    def flush_stdout_stderr(self) -> tuple[str, str]:
        """Flush stdout and stderr"""
        stdout = self._model_stdout.getvalue()
        stderr = self._model_stderr.getvalue()
        self._model_stdout.truncate(0)
        self._model_stderr.truncate(0)
        return stdout, stderr

    def condense_context(self, ctx: list[Document]) -> str:
        """
        Condense the context into a single string
        Args:
            ctx (list[Document]): list of documents
        Returns:
            str: condensed context
        """
        if self._template is None:
            msg = "No template set."
            raise ValueError(msg)
        self._docs = ctx
        logging.debug("Calculating condensed context...")
        template: str = self._template.format(question=self._question, context="")
        template_tokens: int = self._llm.get_num_tokens(template)
        available_tokens: int = self._max_prompt_length - template_tokens
        context: str = "- " + "\n- ".join([d.page_content.strip() for d in ctx])
        context_tok: list[int] = self._llm.client.tokenize(context.encode())
        context_length = len(context_tok)
        if context_length > available_tokens:
            logging.debug("Context too long, truncating...")
            context_tok = context_tok[-available_tokens:]
            bcontext: bytes = self._llm.client.detokenize(context_tok)
            context = bcontext.decode()
        logging.debug("Condensed context calculated")
        return context

    @property
    def last_docs(self) -> list[Document] | None:
        """Get the last docs"""
        return self._docs

    def _handle_command(self, kw: tuple[str | None, str | None, str | None, bool]) -> None:
        """
        Handle the command
        """
        logging.debug(f"Command: {kw[0]}")
        self._chain = self._gen_chain
        if kw[0] in ["redo", "repeat", "again"] and self._last_query is not None:
            kw = extract_keywords(self._last_query, self._known_locations)
            self._question = self._last_query
            self._set_keywords(kw)
            return
        if kw[0] in ["search", "find", "websearch", "google"]:
            self._web_search = True
            self._set_keywords(kw)
            return
        if kw[0] in ["exit", "quit"]:
            self._question = "exit"
            msg = "Exiting..."
            raise KeyboardInterrupt(msg)
        if kw[0] in ["continue", "more", "next"]:
            if self._can_continue():
                self._chain = self._continue_chain
                return
            msg = "No context to continue"
            raise ValueError(msg)

        msg = "Unknown command"
        raise LookupError(msg)

    def _set_keywords(self, kw: tuple[str | None, str | None, str | None, bool]) -> None:
        """
        Set the keywords
        """
        self._keywords["command"] = kw[0]
        self._keywords["where"] = kw[1]
        self._keywords["what"] = kw[2]
        self._keywords["known"] = kw[3]

    def _query_keywords(self) -> None:
        """
        Query the keywords from the question
        Args:
            question (str): user's question
        """
        if self._question is None:
            msg = "No question to query"
            raise ValueError(msg)
        self._web_search = False
        kw = extract_keywords(self._question, self._known_locations)
        try:
            self._handle_command(kw)
            return
        except LookupError:
            logging.debug("Unknown command, using full query")
            self._set_keywords(kw)
            self._last_query = self._question

    @property
    def last_context(self) -> str | None:
        """Get the last context"""
        return self._last_context

    @property
    def last_keywords(self) -> dict[str, str | bool | None]:
        """Get the last keywords"""
        return self._keywords

    @property
    def last_query(self) -> str | None:
        """Get the last query"""
        return self._last_query

    def get_answer(self, question: str) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            question (str): user's question
        Returns:
            str: llm answer
        """
        self._question = question
        self._query_keywords()
        with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
            return self._chain.invoke({"question": self._question})

    def get_answer_stream(self, question: str) -> Iterator[str]:
        """
        Get the answer from llm based on context and user's question
        Args:
            question (str): user's question
        Returns:
            Iterator[str]: llm answer
        """
        self._question = question
        self._query_keywords()
        with store_stdout_stderr(self._model_stdout, self._model_stderr, self._autoflush):
            return self._chain.stream({"question": self._question})

    def response_handler(self) -> ModelResponseHandler:
        """Get the response handler"""
        return self._model_response_handler

    def subscribe(self, subscriber: Callable[[str], None]):
        """Subscribe to updates of the associated property."""
        self._model_response_handler._model_response.subscribe(subscriber)
