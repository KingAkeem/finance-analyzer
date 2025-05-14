import requests

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup

# Initialize local LLM (e.g., Mistral 7B)
llm = Ollama(model="mistral")  # Run `ollama pull mistral` first

def get_headlines() -> list[str]:
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
    prompt = PromptTemplate(
        input_variables=["headlines"],
        template=" ".join([
            "Analyze these financial headlines: {headlines}.",
            "Identify key trends, providek key insights, and summarize the overall sentiment.",
            "Don't repeat the headlines.",
            "Be concise and clear.",
            "Categorize the headlines into sectors and provide a brief summary for each sector such as Technology (can have sub categories such as fintech), Finance, Healthcare, etc.",
            "Always provide a summary of the overall sentiment.",
        ])
    )
    chain = prompt | llm  # LangChain LCEL syntax
    return chain.invoke({"headlines": "\n".join(headlines)})

if __name__ == "__main__":
    headlines = get_headlines()
    print("📰 Headlines\n\n", headlines, "\n")
    analysis = analyze_headlines(headlines)
    print("🔍 Analysis\n\n", analysis, "\n")