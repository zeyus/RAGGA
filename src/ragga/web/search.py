import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import bs4
import requests
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever

if TYPE_CHECKING:
    from langchain_core.callbacks.base import Callbacks
    from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

class Search:
    def __init__(self, max_results: int = 5) -> None:
        self.max_results = max_results
        self._user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
        )
        self._cache_max_size = 100
        self._result_cache: OrderedDict[str, list[Document]] = OrderedDict()

    def _search(self, question: str) -> None:
        logging.info(f"Searching for: {question}")
        search_url = f"https://html.duckduckgo.com/html/?q={question}"
        response = requests.get(search_url, timeout=5, headers={"User-Agent": self._user_agent})
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        self.results = soup.find_all("div", class_="result")

    def get_results(self, question) -> list[Document]:
        if question in self._result_cache:
            self._result_cache.move_to_end(question)
            return self._result_cache[question]
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
        while len(self._result_cache) >= self._cache_max_size:
            self._result_cache.popitem(last=False)
        self._result_cache[question] = results

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
