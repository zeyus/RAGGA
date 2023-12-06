from langchain.docstore.document import Document
from langchain.document_loaders import HuggingFaceDatasetLoader

# this is very temporary
loader = HuggingFaceDatasetLoader("cnn_dailymail", "highlights", name="3.0.0")


test_news = [
    "2023-12-01 - Today something happened.",
]


docs = loader.load()[:10000] # get a sample of news
# add openai news to our list of docs
docs.extend([
    Document(page_content=x) for x in test_news
])

del loader
del test_news
