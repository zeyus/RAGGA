from abc import ABC, abstractmethod
from types import MappingProxyType

from langchain.prompts import ChatMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.prompts import BasePromptTemplate

from ragga.core.config import Config, Configurable

# temporary placeholder...we need local preprocessing
# to handle keywords etc (search? dates?)
simple_template = """
You are {user}'s AI assistant with access to the {user}'s personal notes and internet search.
Use the following context to help answer the {user}'s query at the end.

CONTEXT START

Today is {date}. Here are some excerpts from relevante notes:
{context}

CONTEXT END

Query: {question}
Answer: """

simple_prompt = PromptTemplate.from_template(simple_template)

simple_template_phi2 = """
Instruct: As {user}'s personal assistant, use the following to help answer their question: "{question}"

Relevant personal notes from the user:
{context}


{user}: {question}

Output:"""

simple_prompt_phi2 = PromptTemplate.from_template(simple_template_phi2)


class Prompt(ABC, Configurable):
    """Base prompt"""

    _config_key = "prompt"

    @property
    def user_name(self):
        return self.config[self._config_key]["user_name"]

    @property
    def ai_name(self):
        return self.config[self._config_key]["AI_name"]

    @abstractmethod
    def get_prompt(self) -> BasePromptTemplate:
        """Get the prompt"""
        pass

class ChatPrompt(Prompt):
    """Chat prompt"""

    _default_config = MappingProxyType(
        {
            "user_name": "User",
            "AI_name": "PAI",
            "instruct_user": "Instruct",
            "instructions":
                "You are \"<<AI_NAME>>\", {user}'s personal AI assistant. "
                "Use the following to help answer {user}'s question: \"{question}\"",
            "pre_context": "Relevant personal notes from the user:",
            "post_context": "",
            "user_query": "{question}",
        }
    )

    def __init__(self, conf: Config) -> None:
        super().__init__(conf)
        for k, v in self.config[self._config_key].items():
            if isinstance(v, str):
                self.config[self._config_key][k] = v.replace("<<AI_NAME>>", self.config[self._config_key]["AI_name"])

    def get_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ChatMessagePromptTemplate.from_template(
                role=self.config[self._config_key]["instruct_user"],
                template="\n\n".join([
                    self.config[self._config_key]["instructions"],
                    self.config[self._config_key]["pre_context"],
                    "{context}",
                    self.config[self._config_key]["post_context"],
                ])
            ),
            ChatMessagePromptTemplate.from_template(
                role=self.config[self._config_key]["AI_name"],
                template="How can I help?"
            ),
            ChatMessagePromptTemplate.from_template(
                role=self.config[self._config_key]["user_name"] + " query:",
                template=self.config[self._config_key]["user_query"]
            )
        ])


