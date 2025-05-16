import requests
import re
from typing import List, Dict
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentType
from langchain.agents import initialize_agent
from langchain.schema import SystemMessage

# Initialize LLMs
llm = OllamaLLM(base_url="http://ollama:11434", model="mistral")
research_llm = OllamaLLM(base_url="http://ollama:11434", model="llama3")  # More capable model for research

# Web Scraping Functions (unchanged)
def get_reuters_headlines() -> List[str]:
    headers = {"User-Agent": UserAgent().random, "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.google.com/"}
    response = requests.get("https://www.reuters.com/business/finance/", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        h.get_text(strip=True) 
        for item in soup.find_all("li", class_=re.compile("story-collection__list-item"))
        if (h := item.find(['h2', 'h3', 'a'], {'data-testid': 'Heading'}))
    ]

def get_yahoo_headlines() -> List[str]:
    headers = {"User-Agent": UserAgent().random}
    response = requests.get("https://finance.yahoo.com/topic/latest-news/", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    generic_headlines = {'News', 'Life', 'Entertainment', 'Finance', 'Sports', 'New on Yahoo'}
    return [h.text.strip() for h in soup.find_all("h3") if h.text.strip() not in generic_headlines]

# Research Tools
class FinancialResearchTools:
    @staticmethod
    def search_sec_filings(query: str) -> str:
        """Mock SEC filings search"""
        return f"SEC filings for {query}: Revenue ↑5% QoQ, EPS $1.25"
    
    @staticmethod
    def get_market_sentiment() -> str:
        """Mock sentiment analysis"""
        return "Current market sentiment: Neutral (Fear & Greed Index: 52)"

tools = [
    Tool(
        name="SECFilings",
        func=FinancialResearchTools.search_sec_filings,
        description="Search SEC filings for company financial data"
    ),
    Tool(
        name="MarketSentiment",
        func=FinancialResearchTools.get_market_sentiment,
        description="Get current market sentiment indicators"
    )
]

# Analysis Pipeline
def analyze_and_research():
    # Step 1: Get headlines
    headlines = [*get_yahoo_headlines(), *get_reuters_headlines()]
    print("📰 Headlines\n", "\n- ".join(headlines), "\n")
    
    # Step 2: Initial analysis
    analysis_prompt = PromptTemplate.from_template("""
    Analyze these financial headlines and recommend specific research steps:
    {headlines}
    
    Focus on:
    1. Key companies mentioned
    2. Potential market impacts
    3. Required data verification
    
    Format recommendations as:
    - Research [topic] using [tool/method]
    - Verify [claim] by checking [source]
    """)
    
    recommendations = (analysis_prompt | llm).invoke({"headlines": "\n".join(headlines)})
    print("🔍 Analysis Recommendations\n", recommendations, "\n")
    
    # Step 3: Conduct research
    research_agent = initialize_agent(
        tools,
        research_llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        agent_kwargs={
            "system_message": SystemMessage(content="""
            Financial Research Assistant Guidelines:
            1. Always verify facts using tools
            2. Cite sources for all data
            3. Distinguish between facts and interpretations
            """)
        },
        verbose=True
    )
    
    research_report = research_agent.run(
        f"Conduct research based on these recommendations:\n{recommendations}\n"
        f"Original headlines:\n{headlines}"
    )
    
    print("📊 Final Research Report\n", research_report)

if __name__ == "__main__":
    analyze_and_research()