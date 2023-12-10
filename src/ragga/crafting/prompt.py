from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

# temporary placeholder...we need local preprocessing
# to handle keywords etc (search? dates?)
simple_template = """
You are an AI assistant with access to the user's personal notes and internet search.
Use the following context to help answer the user's query at the end.

CONTEXT START

Today is {date}. Here are some excerpts from relevante notes:
{context}

CONTEXT END

Query: {question}
Answer: """

simple_prompt = PromptTemplate.from_template(simple_template)


# QA_TEMPLATE = """{preamble}

# {rag_context}

# {qa}"""

# qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)

# preamble_template = "Use the following pieces of context to answer the question at the end taking in consideration the dates:"
# preamble_prompt = PromptTemplate.from_template(preamble_template)

# context_template = """
# {context}
# """
# context_prompt = PromptTemplate.from_template(context_template)

# question_template = """Question: {question}
# Answer: """
# question_prompt = PromptTemplate.from_template(question_template)

# input_prompts = [
#     ("preamble", preamble_prompt),
#     ("rag_context", context_prompt),
#     ("qa", question_prompt),
# ]

# pipeline_prompt = PipelinePromptTemplate(
#     final_prompt = qa_prompt,
#     pipeline_prompts = input_prompts
# )
