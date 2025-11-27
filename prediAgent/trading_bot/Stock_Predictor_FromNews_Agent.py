import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from google import genai
from google.genai import types
import os

# optional: import genai if available in your environment
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Load your API keys from env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
NEWS_API_KEY = os.getenv("YOUR_NEWS_API_KEY")  # e.g. MarketAux, NewsAPI, etc.

# Initialize Gemini client if key provided; otherwise run in non-LLM fallback mode
client = None
if GOOGLE_API_KEY:
    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
    except Exception:
        client = None

def fetch_historical_news(ticker: str, from_date: str, to_date: str, limit: int = 50):
    """
    Example for MarketAux. Modify if using another API.
    """
    # If a third-party news API key is configured, use it (example MarketAux).
    if NEWS_API_KEY:
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            "api_token": NEWS_API_KEY,
            "symbols": ticker,
            "published_after": from_date,
            "published_before": to_date,
            "limit": limit
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json().get("data", [])

    # Fallback: use yfinance news (no date filtering available; return recent items)
    try:
        t = yf.Ticker(ticker)
        raw = getattr(t, "news", [])
        out = []
        for item in raw[:limit]:
            # try to normalize keys to match MarketAux-like structure
            published = item.get('providerPublishTime') or item.get('pubDate') or item.get('datetime')
            try:
                if isinstance(published, (int, float)):
                    published = pd.to_datetime(int(published), unit='s', utc=True).isoformat()
                elif published is None:
                    published = None
            except Exception:
                published = None

            out.append({
                'published_at': published,
                'title': item.get('title') or item.get('headline') or '',
                'description': item.get('summary') or item.get('content') or '',
                'url': item.get('link') or item.get('canonicalLink') or ''
            })
        return out
    except Exception:
        return []

def fetch_latest_news(ticker: str, limit: int = 5):
    # This is a simple example: you might hit "recent" news endpoint or change dates
    # pick a recent window (last 30 days) for fallback
    end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d')
    return fetch_historical_news(ticker, start, end, limit=limit)

def build_prompt(ticker: str, latest_articles, historical_articles):
    prompt = f"""You are a financial analyst. Based on the following news for **{ticker}**:

Latest News:
"""
    for art in latest_articles:
        prompt += f"- [{art['published_at']}] {art['title']}: {art.get('description', '')}\n"

    prompt += "\nHistorical News:\n"
    for art in historical_articles:
        prompt += f"- [{art['published_at']}] {art['title']}: {art.get('description','')}\n"

    prompt += (
        "\nGiven this information, predict the **short-term (1‑5 years) price movement** of the stock:\n"
        "- up\n- down\n- neutral\n\n"
        "Give your prediction and a one-sentence justification."
    )

    prompt += (
    "\nFirst, think step-by-step about the key drivers, risks, and how the news may impact the company's business or stock. "
    "Then, provide a prediction for the short-term (1-3 months) price movement of the stock. "
    "Your prediction must be exactly one of the following:\n"
    "- up\n"
    "- down\n"
    "- neutral\n\n"
    "Finally, give a one-sentence justification for your prediction."
)

    print("Length of prompt is :::::",len(prompt))
    return prompt

def build_reasoned_prompt(ticker: str, latest, historical):
    prompt = (
        f"You are a seasoned financial analyst. Analyze the following news for **{ticker}**:\n\n"
        "Latest News:\n"
    )
    for art in latest:
        prompt += f"- [{art.get('published_at')}] {art.get('title')}: {art.get('description', '')}\n"

    prompt += "\nHistorical News:\n"
    for art in historical:
        prompt += f"- [{art.get('published_at')}] {art.get('title')}: {art.get('description', '')}\n"

    # Ask for step-by-step reasoning
#     prompt += (
#     "\nFirst, think step-by-step about the key drivers, risks, and how the news may impact the company's business or stock. "
#     "Then, provide a prediction for the short-term (1-3 months) price movement of the stock. "
#     "Your prediction must be exactly one of the following:\n"
#     "- up\n"
#     "- down\n"
#     "- neutral\n\n"
#     "Finally, give a one-sentence justification for your prediction."
# )
    prompt += (
        "\nYour prediction must be exactly one of the following:\n"
        "- up\n"
        "- down\n"
        "- neutral\n\n"
        "Finally, give a one-sentence justification for your prediction."
        "Give your prediction and a one-sentence justification."
    )
    return prompt

def predict_ticker_from_news(ticker: str):
    latest = fetch_latest_news(ticker)
    hist = fetch_historical_news(ticker, "2020-01-01", "2025-10-01", limit=30)

    prompt = build_reasoned_prompt(ticker, latest, hist)

    # Call Gemini
    if client is None:
        # No LLM available: return the assembled prompt as a fallback diagnostic
        return "LLM not configured; returning prompt for local inspection:\n\n" + prompt[:4000]

    
    resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=400)
    )
    #print("Response text:", resp.text)
    return resp.text


if __name__ == "__main__":
    ticker = "NVDA"
    pred = predict_ticker_from_news(ticker)
    print(ticker , ":Prediction:", pred)