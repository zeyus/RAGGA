import re
import typing as t
from operator import itemgetter

import yaml
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.document_transformers import (
    EmbeddingsClusteringFilter,
    EmbeddingsRedundantFilter,
    LongContextReorder,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompt_values import PromptValue







QA_TEMPLATE = """{preamble}

{rag_context}

{qa}"""

qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

preamble_template = "Use the following pieces of context to answer the question at the end taking in consideration the dates:"
preamble_prompt = PromptTemplate.from_template(preamble_template)

context_template = """
{context}
"""
context_prompt = PromptTemplate.from_template(context_template)

question_template = """Question: {question}
Answer: """
question_prompt = PromptTemplate.from_template(question_template)

input_prompts = [
    ("preamble", preamble_prompt),
    ("rag_context", context_prompt),
    ("qa", question_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt = qa_prompt,
    pipeline_prompts = input_prompts
)

simple_template = """
Use the following pieces of context to answer the question at the end taking in consideration the dates:
{context}


Question: {question}
Answer: """

simple_prompt = PromptTemplate.from_template(simple_template)

print(pipeline_prompt.input_variables)



openai_news = [
    "2023-11-22 - Sam Altman returns to OpenAl as CEO with a new initial board of Bret Taylor (Chair), Larry Summers, and Adam D'Angelo.",
    "2023-11-21 - Ilya and the board's decision to fire Sam from OpenAI caught everyone off guard, with no prior information shared.",
    "2023-11-21 - In a swift response, Sam was welcomed into Microsoft by Satya Nadella himself.",
    "2023-11-21 - Meanwhile, a staggering 500+ OpenAI employees made a bold move, confronting the board with a letter: either step down or they will defect to Sam's new team at Microsoft.",
    "2023-11-21 - In a jaw-dropping twist, Ilya, integral to Sam's firing, also put his name on that very letter. Talk about an unexpected turn of events!",
    "2023-11-20 - BREAKING: Sam Altman and Greg Brockman Join Microsoft, Emmett Shear Appointed CEO of OpenAI",
    "2023-11-20 - Microsoft CEO Satya Nadella announced a major shift in their partnership with OpenAI. Sam Altman and Greg Brockman, key figures at OpenAI, are now joining Microsoft to lead a new AI research team. This move marks a significant collaboration and potential for AI advancements. Additionally, Emmett Shear, former CEO of Twitch, has been appointed as the new CEO of OpenAI, signaling a new chapter in AI leadership and innovation.",
    "2023-11-20 - Leadership Shakeup at OpenAI - Sam Altman Steps Down!",
    "2023-11-20 - Just a few days after presenting at OpenAI's DevDay, CEO Sam Altman has unexpectedly departed from the company, and Mira Murati, CTO of the company, steps in as Interim CEO. This is a huge surprise and speaks volumes about the dynamic shifts in tech leadership today.",
    """2023-11-20 - What's Happening at OpenAI?
    - Sam Altman, the face of OpenAI, is leaving not just the CEO role but also the board of directors.
    - Mira Murati, an integral part of OpenAI's journey and a tech visionary, is taking the helm as interim CEO.
    - The board is now on a quest to find a permanent successor.""",
    "2023-11-20 - The transition raises questions about the future direction of OpenAI, especially after the board's statement about losing confidence in Altman's leadership.",
    """2023-11-20 - With a board consisting of AI and tech experts like Ilya Sutskever, Adam D'Angelo, Tasha McCauley, and Helen Toner, OpenAI is poised to continue its mission. Can they do it without Sam?
    - Greg Brockman, stepping down as chairman, will still play a crucial role, reporting to the new CEO."""
]
loader = HuggingFaceDatasetLoader("cnn_dailymail", "highlights", name='3.0.0')
docs = loader.load()[:10000] # get a sample of news
# add openai news to our list of docs
docs.extend([
    Document(page_content=x) for x in openai_news
])

encoder = Encoder()
faiss_db = VectorDatabase()


passages = faiss_db.create_passages_from_documents(docs)
faiss_db.store_passages_db(passages, encoder.encoder)

generator = Generator(simple_prompt, faiss_db, encoder)

QUERY = "What happened to the CEO of OpenAI?"
# QUERY = "What is happening in Israel?"
print(QUERY)
print("Answer:")
result = generator.get_answer(QUERY)



while True:
    try:
        query = input("Ask a question: ")
        print("Answer:")
        result = generator.get_answer(query)
        # print(chunk, end='', flush=True)
        # print("", flush=True)
        print(result)
    except KeyboardInterrupt:
        print("Bye!")
        break
    except Exception as e:
        print("Exception occurred. Quitting, bye!")
        print(e)
        break
