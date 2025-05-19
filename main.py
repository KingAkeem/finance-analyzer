import requests
import re
import logging

from pydantic import BaseModel, Field
from edgar import Company, TXTML
from typing import List
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.schema import SystemMessage
from langchain.tools import StructuredTool

# Initialize LLMs
llm = OllamaLLM(base_url="http://ollama:11434", model="mistral")
research_llm = OllamaLLM(base_url="http://ollama:11434", model="llama3")  # More capable model for research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Web Scraping Functions (unchanged)
def get_reuters_headlines() -> List[str]:
    headers = {"User-Agent": UserAgent().random, "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.google.com/"}
    logging.info("Fetching Reuters headlines...")
    response = requests.get("https://www.reuters.com/business/finance/", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [
        h.get_text(strip=True) 
        for item in soup.find_all("li", class_=re.compile("story-collection__list-item"))
        if (h := item.find(['h2', 'h3', 'a'], {'data-testid': 'Heading'}))
    ]
    logging.info(f"Fetched {len(headlines)} Reuters headlines")
    return headlines

def get_yahoo_headlines() -> List[str]:
    headers = {"User-Agent": UserAgent().random}
    logging.info("Fetching Yahoo headlines...")
    response = requests.get("https://finance.yahoo.com/topic/latest-news/", headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    generic_headlines = {'News', 'Life', 'Entertainment', 'Finance', 'Sports', 'New on Yahoo'}
    headlines = [h.text.strip() for h in soup.find_all("h3") if h.text.strip() not in generic_headlines]
    logging.info(f"Fetched {len(headlines)} Yahoo headlines")
    return headlines

def get_cik_from_ticker(ticker):
    headers = {
        "User-Agent": "Akeem King (akeemtlking@gmail.com)",  # SEC requires this
        "Accept-Encoding": "gzip, deflate",
    }
    url = f"https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers).json()
    for entry in response.values():
        if entry["ticker"] == ticker.upper():
            logging.info(f"Found CIK for ticker {ticker}: {entry['cik_str']}")
            return str(entry["cik_str"]).zfill(10)  # CIK must be 10 digits
    raise ValueError(f"Ticker {ticker} not found in SEC database")


def company_to_ticker(company_name: str) -> str:
    """
    Convert common company names to their stock ticker symbols.
    
    Args:
        company_name (str): The name of the company to convert
        
    Returns:
        str: The stock ticker symbol or the original name if not found
    """
    ticker_mapping = {
        # Technology
        'microsoft': 'MSFT',
        'apple': 'AAPL',
        'nvidia': 'NVDA',
        'amazon': 'AMZN',
        'alphabet': 'GOOGL',  # Google's parent company
        'google': 'GOOGL',
        'meta': 'META',       # Facebook's parent company
        'facebook': 'META',
        'tesla': 'TSLA',
        'intel': 'INTC',
        'amd': 'AMD',
        'adobe': 'ADBE',
        'salesforce': 'CRM',
        'oracle': 'ORCL',
        'cisco': 'CSCO',
        'ibm': 'IBM',
        'netflix': 'NFLX',
        
        # Retail & Consumer
        'walmart': 'WMT',
        'target': 'TGT',
        'home depot': 'HD',
        'costco': 'COST',
        'mcdonald\'s': 'MCD',
        'starbucks': 'SBUX',
        'nike': 'NKE',
        'disney': 'DIS',
        
        # Financial
        'jpmorgan chase': 'JPM',
        'bank of america': 'BAC',
        'wells fargo': 'WFC',
        'goldman sachs': 'GS',
        'morgan stanley': 'MS',
        'visa': 'V',
        'mastercard': 'MA',
        
        # Healthcare
        'johnson & johnson': 'JNJ',
        'pfizer': 'PFE',
        'moderna': 'MRNA',
        'merck': 'MRK',
        
        # Energy & Industrials
        'exxon mobil': 'XOM',
        'chevron': 'CVX',
        'boeing': 'BA',
        'general electric': 'GE',
        
        # Telecom
        'verizon': 'VZ',
        'at&t': 'T',
        
        # Other well-known companies
        'coca cola': 'KO',
        'pepsi': 'PEP',
        'procter & gamble': 'PG',
        '3m': 'MMM'
    }
    
    # Convert to lowercase and strip whitespace for matching
    normalized_name = company_name.lower().strip()
    
    return ticker_mapping.get(normalized_name, company_name.strip())


# Research Tools
class FinancialResearchTools:
    @staticmethod
    def search_sec_filings(company_or_ticker: str) -> str:
        """Mock SEC filings search"""
        try:
            # Convert to ticker if needed
            ticker = company_to_ticker(company_or_ticker)
            cik = get_cik_from_ticker(ticker)
            
            # Get 10-K filing
            doc = Company(name=ticker, cik=cik).get_10K()
            filing_text = TXTML.parse_full_10K(doc)
            
            # Analysis prompt
            prompt = f"""Analyze this SEC 10-K filing for {ticker} and provide:
            
            === Financial Highlights ===
            - Revenue growth trends
            - Profitability metrics (EPS, ROE, ROA)
            - Key financial ratios
            
            === Risk Factors ===
            - Notable red flags
            - Significant risks mentioned
            
            === Business Insights ===
            - Major business developments
            - Strategic initiatives
            
            Filing Excerpt:
            {filing_text[:15000]}... [truncated]
            """
            
            return research_llm.invoke(prompt)
            
        except Exception as e:
            return f"Error analyzing {company_or_ticker}: {str(e)}"
    
    @staticmethod
    def get_market_sentiment() -> str:
        """Mock sentiment analysis"""
        return "Current market sentiment: Neutral (Fear & Greed Index: 52)"

class SECFilingsInput(BaseModel):
    ticker: str = Field(
        description="Ticker symbol for SEC filings search",
        example="AAPL"
    )

tools = [
    StructuredTool.from_function(
        name="SECFilings",
        func=FinancialResearchTools.search_sec_filings,
        description="Search SEC filings for company financial data. Input one company name or ticker symbol at a time.",
        args_schema=SECFilingsInput
    ),
    StructuredTool.from_function(
        name="MarketSentiment",
        func=FinancialResearchTools.get_market_sentiment,
        description="Get current market sentiment indicators",
    )
]

research_agent = initialize_agent(
    tools,
    research_llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={
        "system_message": SystemMessage(content="""
        You are a financial research assistant. Follow these rules:
        1. Always verify facts using tools
        2. For company data, use SECFilings with either:
           - Company name (e.g., "Apple")
           - Ticker symbol (e.g., "AAPL")
        3. Present findings in clear, organized text
        """)
    },
    verbose=True
)

# Analysis Pipeline
def analyze_and_research():
    # Step 1: Get headlines
    logging.info("Fetching headlines...")
    headlines = [*get_yahoo_headlines(), *get_reuters_headlines()]
    print("\n🔍 Analyzing headlines...")
    analysis = llm.invoke(
        f"Analyze these financial headlines and suggest research priorities:\n"
        f"{chr(10).join(headlines)}\n\n"
        "Focus on:\n"
        "- Companies needing deeper analysis\n"
        "- Potential market impacts\n"
        "- Controversial topics"
    )
    
    print("\n📊 Conducting research...")
    research_report = research_agent.run(
        f"Based on this analysis, conduct thorough research:\n\n"
        f"Headlines:\n{chr(10).join(headlines[:5])}\n\n"
        f"Analysis:\n{analysis}"
    )
    
    print("\n📝 Final Report:")
    print(research_report)

if __name__ == "__main__":
    analyze_and_research()