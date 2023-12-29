import logging
from typing import TYPE_CHECKING, Any

import bs4
import requests
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever

if TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

class Search:
    def __init__(self, max_results=5):
        self.max_results = max_results
        self._user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
        )

    def _search(self, question):
        logging.info(f"Searching for: {question}")
        search_url = f"https://html.duckduckgo.com/html/?q={question}"
        response = requests.get(search_url, timeout=5, headers={"User-Agent": self._user_agent})
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        self.results = soup.find_all("div", class_="result")

    def get_results(self, question) -> list[Document]:
        self._search(question)
        results = []
        for result in self.results[: self.max_results]:
            title = result.find("a", class_="result__a").text
            url = result.find("a", class_="result__a")["href"]
            snippet = result.find("a", class_="result__snippet").text
            results.append(
                Document(
                    page_content=title + "\n" + snippet,
                    metadata={"url": url}
                )
            )
        return results


class SearchRetriever(BaseRetriever):
    search: Search

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(
        self,
        query: str,
        *,
        _callbacks: "Callbacks" = None,
        _tags: list[str] | None = None,
        _metadata: dict[str, Any] | None = None,
        _run_name: str  | None = None,
        **_kwargs: Any,
    ) -> list[Document]:
        return self.search.get_results(query)

    def _get_relevant_documents(
        self,
        query: str, *,
        run_manager: "CallbackManagerForRetrieverRun"  # noqa: ARG002
    ) -> list[Document]:
        return self.search.get_results(query)


WebSearchRetriever = SearchRetriever(search=Search())
