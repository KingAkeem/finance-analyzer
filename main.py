import requests
import re

from langchain_ollama import OllamaLLM  # Updated import
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# Initialize local LLM (e.g., Mistral 7B)
llm = OllamaLLM(
    base_url="http://ollama:11434",
    model="mistral"
)

def get_reuters_headlines() -> list[str]:
    """
    Fetch the latest financial headlines from Yahoo Finance.
    Returns:
        list[str]: List of financial headlines.
    """
    headers = {
        "User-Agent": UserAgent().random,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    response = requests.get("https://www.reuters.com/business/finance/", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    news_item_elements = soup.find_all("li", class_=re.compile("story-collection__list-item"))
    headlines = []
    for item in news_item_elements:
        headline = item.find(['h2', 'h3', 'a'], {'data-testid': 'Heading'})
        if headline:
            headlines.append(headline.get_text(strip=True))
    return headlines


def get_yahoo_headlines() -> list[str]:
    """
    Fetch the latest financial headlines from Yahoo Finance.
    Returns:
        list[str]: List of financial headlines.
    """
    # Use a random user agent to avoid bot detection
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.4; en-US; rv:1.9.2.2) Gecko/20100316 Firefox/3.6.2"}
    # Parse the Yahoo Finance page
    response = requests.get("https://finance.yahoo.com/topic/latest-news/", headers=headers)


    soup = BeautifulSoup(response.text, "html.parser")
    # filter out generic headlines, they are not useful for analysis
    generic_headlines = ['News', 'Life', 'Entertainment', 'Finance', 'Sports', 'New on Yahoo']
    return [h.text.strip() for h in soup.find_all("h3") if h.text.strip() not in generic_headlines]


def analyze_headlines(headlines: list[str]) -> str:
    """
    Analyze the financial headlines using a local LLM.
    Args:
        headlines (list[str]): List of financial headlines.
    Returns: 
        str: Analysis of the headlines.
    """
    template=f"""
    As a cautious financial analyst working with limited headline data, provide:
    1) Observable patterns
    2) Plausible interpretations
    3) Required verification steps
    
    Headlines:
    {headlines}
    
    Structure your response with:
    - Key Observations
    - Tentative Conclusions
    - Critical Unknowns
    - Recommended Next Steps
    
    Always use hedging language like "may suggest" or "could indicate".
    """
    prompt = PromptTemplate(
        input_variables=["headlines"],
        template=template
    )
    chain = prompt | llm  # LangChain LCEL syntax
    return chain.invoke({"headlines": "\n".join(headlines)})

if __name__ == "__main__":
    headlines = [*get_yahoo_headlines(), *get_reuters_headlines()]
    print("📰 Headlines\n\n", headlines, "\n")
    analysis = analyze_headlines(headlines)
    print("🔍 Analysis\n\n", analysis, "\n")