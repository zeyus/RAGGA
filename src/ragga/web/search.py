import bs4
import requests


class Search:
    def __init__(self, max_results=5):
        self.max_results = max_results
        self._user_agent = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
        )

    def _search(self, question):
        search_url = f"https://html.duckduckgo.com/html/?q={question}"
        response = requests.get(search_url, timeout=5, headers={"User-Agent": self._user_agent})
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        self.results = soup.find_all("div", class_="result")

    def get_results(self, question):
        self._search(question)
        results = []
        for result in self.results[: self.max_results]:
            title = result.find("a", class_="result__a").text
            url = result.find("a", class_="result__a")["href"]
            snippet = result.find("a", class_="result__snippet").text
            results.append({"title": title, "url": url, "snippet": snippet})
        return results
