import json
import re

from edgar import Company, TXTML, Edgar
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

llm = OllamaLLM(base_url="http://localhost:11434", model="llama3")  # More capable model for research

# Find the company
e = Edgar()
company = "INTERNATIONAL BUSINESS MACHINES CORP"
cik = e.get_cik_by_company_name(name=company)  # This will return the CIK for IBM
company = Company(company, cik)

# Parse the latest 10-K filing
doc = company.get_10K()
text = TXTML.parse_full_10K(doc)
soup = BeautifulSoup(text, parser="lxml", features="lxml")

# Analyze the 10-K filing
analysis_prompt = PromptTemplate.from_template("""
Analyze {company} performance based off the follwing SEC 10K filing:
{filings}
Provide valid JSON output with the following information:
    - Include any relevant financial ratios or metrics that are commonly used in investment analysis.
    - Highlight any potential red flags or areas of concern that investors should be aware of.
    - Include all the releavnt quantitative and qualitative data points along with temporal information.
    - Only return JSON data, do not include any other text or explanation.

And the structure of the JSON should be as follows:
 - "Company" 
    - "Name": "string",
    - "Ticker": "string"
- "Financials": [
    - "Year": "number",
    - "Revenue": "string (with unit)",
    - "Net Income": "string (with unit)",
    - "EPS": "string (with unit)",
    - "ROE": "string (%)",
    - "ROA": "string (%)"
  ],
- "Red Flags": [
    - "Year": "number",
    - "Segment": "string",
    - "Issue (this will be the acutal name of the issue)": "string (with metric)"
  ],
- "Qualitative Data": [
    - "Year": "number",
    - "Description": "string"
  ]
```
""")
sec_filing_analysis = (analysis_prompt | llm).invoke({"filings": soup.text, "company": company})

def extract_and_parse_json(json_string):
    """Extracts JSON from a string containing mixed content and parses it"""
    # Use regex to find the JSON portion (between ```json and ```)
    json_match = re.search(r'```json\n(.*?)\n```', json_string, re.DOTALL)
    
    if not json_match:
        raise ValueError("No JSON found in the string")
    
    json_str = json_match.group(1).strip()
    return json.loads(json_str)

print(sec_filing_analysis, "\n")
print(extract_and_parse_json(sec_filing_analysis))