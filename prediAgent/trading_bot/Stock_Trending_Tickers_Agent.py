import requests
from bs4 import BeautifulSoup

from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd
import os
from typing import Optional
import smtplib
from smtplib import SMTPAuthenticationError, SMTPConnectError, SMTPException
from email.mime.text import MIMEText




# optional: import genai if available in your environment
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()



def get_current_stock_price(ticker_symbol: str) -> float | str:
    """
    Fetches the most recent stock price for a given ticker symbol.

    Args:
        ticker_symbol: The stock ticker symbol (e.g., "AAPL", "GOOG").

    Returns:
        The current stock price as a float, or an error message string if not found.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch the most recent data for the last day
        todays_data = ticker.history(period='1d')
        
        if todays_data.empty:
            return f"No data found for ticker {ticker_symbol}. It might be delisted or invalid."
            
        # Return the last closing price
        return todays_data['Close'].iloc[-1]
        
    except Exception as e:
        return f"An error occurred while fetching the price for {ticker_symbol}: {e}"


def get_trending_tickers():
    """Gets a list of trending stock tickers. Use for queries about 'trending stocks', 'hot stocks', or 'popular tickers'.
    
    Returns:
        A list of stock ticker symbols that are currently trending.
    """
    url = "https://www.stockmarketwatch.com/markets/pre-market/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Find the table containing the tickers
        table = soup.find("table", {"class": "tbldata"})  # Replace "datatable" with the actual class name
        
        # Extract tickers from the table rows
        tickers = []
        if table:
            rows = table.find_all("tr")[1:]  # Skip the header row
            for row in rows:
              
                columns = row.find_all("td")
                if columns:
                    volume_text = columns[4].text.strip().upper() # E.g., "1.54M"
                    #print("Volume Text:", volume_text)  # For debugging
                    try:
                        numeric_volume = 0
                        if 'M' in volume_text:
                            numeric_volume = float(volume_text.replace('M', '')) * 1_000_000
                        elif 'K' in volume_text:
                            numeric_volume = float(volume_text.replace('K', '')) * 1_000
                        else:
                            numeric_volume = float(volume_text)

                        #print("Parsed Volume:", numeric_volume)  # For debugging
                        # Only include tickers with significant volume (e.g., > 100,000)
                        if numeric_volume > 100000:
                            ticker = columns[2].text.strip()
                            tickers.append(ticker)
                    except (ValueError, IndexError):
                        # Could not parse volume, skip this row
                        continue       
        return tickers
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return []


def get_news_for_tickers(tickers: list[str]) -> str:
    """
    Fetches the latest news for a list of stock tickers.

    Args:
        tickers: A list of stock ticker symbols (e.g., ['AAPL', 'GOOG']).

    Returns:
        A formatted string containing the news for the tickers, or a message if no news is found.
    """
    import yfinance as yf
    all_news = []
    for ticker_symbol in tickers:
        try:
            ticker = yf.Ticker(ticker_symbol)
            news = ticker.news
            #print(f"DEBUG: News for {ticker_symbol}: {news}")  # Debugging line to inspect news structure
            if news:
                all_news.append(f"--- News for {ticker_symbol.upper()} ---")
                for item in news[:5]:
                    content = item.get('content', {})
                    if not content:
                        all_news.append("  - Malformed news item: No 'content' key found.")
                        continue

                    title = content.get('title', 'No Title Available')
                    summary = content.get('summary', 'No Summary Available')
                    
                    # Safely get the nested URL
                    canonical_url = content.get('canonicalUrl', {})
                    link = canonical_url.get('url', 'No Link Available') if isinstance(canonical_url, dict) else 'No Link Available'

                    all_news.append(f"  - Title: {title}")
                    all_news.append(f"    Summary: {summary}")
                    all_news.append(f"    Link: {link}")
                    all_news.append("") # Add a blank line for readability
        except Exception as e:
            all_news.append(f"Could not fetch news for {ticker_symbol}: {e}")

    if not all_news:
        return "No news found for the provided tickers."

    return "\n".join(all_news)


    print(get_news_for_tickers(['AAPL', 'MSFT']))

import os
import re
import textwrap

# Ensure GENAI_AVAILABLE is defined earlier (True if `from google import genai` succeeded)
try:
    GENAI_AVAILABLE
except NameError:
    GENAI_AVAILABLE = False

def predict_price_from_news(ticker: str, news_summary: str, debug: bool = False) -> str:
    """
    Analyze sentiment of `news_summary` for `ticker` and predict short-term movement:
    'up', 'down', or 'neutral' with a one-sentence justification.

    Returns a single string with the prediction and justification, or an error message.
    """
    # Basic validation
    if not ticker or not str(ticker).strip():
        return "Error: ticker is empty."
    if not news_summary or not str(news_summary).strip():
        return "Error: news_summary is empty."

    if not GENAI_AVAILABLE:
        return "Error: Google GenAI library not available for prediction."

    # Trim excessively long summaries to avoid token limits
    MAX_CHARS = 8000
    if len(news_summary) > MAX_CHARS:
        if debug:
            print(f"[predict] Trimming news_summary from {len(news_summary)} to {MAX_CHARS} chars.")
        news_summary = news_summary[:MAX_CHARS] + "\n\n...TRUNCATED..."

    # Build safe prompt
    prompt = textwrap.dedent(
        f"""
        You are a concise financial analyst. Based ONLY on the short news summary below,
        predict whether the short-term price movement for the ticker {ticker.upper()} is:
          - up
          - down
          - neutral

        Provide your answer on a single line in the following JSON format (no extra text):
        {{
          "prediction": "up|down|neutral",
          "justification": "<one-sentence justification>"
        }}

        News Summary:
        \"\"\"
        {news_summary}
        \"\"\"
        """
    ).strip()

    # Read API key (support two common env names)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment."

    try:
        # Two common GenAI SDK shapes:
        # 1) genai.Client(api_key=...).models.generate_content(...)
        # 2) genai.GenerativeModel(...).generate_content(...)
        resp = None
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
        elif hasattr(genai, "GenerativeModel"):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
            resp = model.generate_content(prompt)
        else:
            return "Error: Unrecognized genai SDK (no Client or GenerativeModel)."

        if debug:
            print("--- DEBUG: raw response ---")
            try:
                print(resp)
            except Exception:
                print(repr(resp))
            print("--------------------------")

        # Extract text from common response shapes
        text_out = None
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            text_out = resp.text.strip()
        # check structured output shapes
        if not text_out:
            # try resp.output -> content -> text
            out = getattr(resp, "output", None)
            if out and isinstance(out, (list, tuple)) and len(out) > 0:
                first = out[0]
                content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
                if content and isinstance(content, (list, tuple)) and len(content) > 0:
                    text_candidate = getattr(content[0], "text", None) or (content[0].get("text") if isinstance(content[0], dict) else None)
                    if text_candidate:
                        text_out = text_candidate.strip()
        if not text_out and hasattr(resp, "content"):
            try:
                parts = resp.content
                if isinstance(parts, (list, tuple)) and parts:
                    texts = []
                    for p in parts:
                        t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                        if t:
                            texts.append(t.strip())
                    if texts:
                        text_out = "\n".join(texts)
            except Exception:
                pass
        if not text_out:
            text_out = str(resp).strip()

        if not text_out:
            return "Prediction failed: API returned empty response."

        # Try to parse JSON if model followed the instruction
        json_match = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
        if json_match:
            try:
                import json
                parsed = json.loads(json_match.group(0))
                pred = parsed.get("prediction")
                just = parsed.get("justification") or parsed.get("reason") or ""
                if pred and pred.lower() in ("up", "down", "neutral"):
                    return f"{pred.lower()}: {just.strip()}"
                # if prediction present but not exactly matching, still return full parsed result
                return json.dumps(parsed)
            except Exception:
                # fall through to text parsing
                pass

        # Fallback: look for keywords in the text
        lowered = text_out.lower()
        if "up" in lowered and "down" not in lowered:
            # try to extract a short justification sentence
            sent_match = re.search(r"([^.]{20,200}\.)", text_out)
            justification = sent_match.group(0).strip() if sent_match else text_out.split("\n")[0].strip()
            return f"up: {justification}"
        if "down" in lowered and "up" not in lowered:
            sent_match = re.search(r"([^.]{20,200}\.)", text_out)
            justification = sent_match.group(0).strip() if sent_match else text_out.split("\n")[0].strip()
            return f"down: {justification}"
        # if neither clearly present, return the model text as-is (useful for debugging)
        return f"neutral (uncertain): {text_out.splitlines()[0][:400]}"

    except Exception as e:
        return f"An error occurred while generating the prediction: {type(e).__name__}: {e}"


import os
import re
import textwrap

# Ensure GENAI_AVAILABLE is defined earlier (True if `from google import genai` succeeded)
try:
    GENAI_AVAILABLE
except NameError:
    GENAI_AVAILABLE = False



def summarize_news_content_old(text_to_summarize: str) -> str:
    """
    Uses a large language model to summarize a block of text into key bullet points.

    Args:
        text_to_summarize: The text content to be summarized.

    Returns:
        A concise summary of the text.
    """
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Error: GOOGLE_API_KEY not found in .env file for summarization."
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-pro')

        prompt = (
            "Summarize the following news article text into 2-3 key bullet points. "
            "Focus on the financial implications, market sentiment, and any specific data points mentioned."
            f"\n\nArticle Text: \"{text_to_summarize}\""
            "\n\nSummary (as bullet points):"
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"An error occurred during summarization: {e}"

import os
import json
import textwrap
from typing import Optional

# assumes GENAI_AVAILABLE flag is defined earlier (True if `from google import genai` succeeded)
# if not defined, set default:
try:
    GENAI_AVAILABLE
except NameError:
    GENAI_AVAILABLE = False

def summarize_news_content(text_to_summarize: str, debug: bool = False) -> str:
    """
    Summarize `text_to_summarize` into 2-3 bullet points using Google GenAI (Gemini).
    Returns a string (summary) or an error message string.

    This function:
    - Validates input,
    - Uses whichever genai API shape is available (client.models.generate_content or GenerativeModel),
    - Safely inspects the response for common fields,
    - Trims very long input to avoid token limits (but warns in debug).
    """
    if not GENAI_AVAILABLE:
        return "Error: Google GenAI library not available for summarization."

    if not text_to_summarize or not text_to_summarize.strip():
        return "Error: Input text for summarization is empty."

    # keep prompt reasonably small to avoid token limits (tune as needed)
    MAX_CHARS = 10000
    if len(text_to_summarize) > MAX_CHARS:
        if debug:
            print(f"[summarize] Input length {len(text_to_summarize)} > {MAX_CHARS}. Trimming.")
        text_to_summarize = text_to_summarize[:MAX_CHARS] + "\n\n...TRUNCATED..."

    # Construct safe prompt (use triple quotes to avoid escaping issues)
    prompt = textwrap.dedent(
        f"""
        Summarize the following news article text into 2-3 concise bullet points.
        Focus on: financial implications, market sentiment, and any key data points.
        
        Article Text:
        \"\"\"
        {text_to_summarize}
        \"\"\"
        
        Summary (2-3 bullet points):
        """
    ).strip()

    # get configured API key (choose the one you actually set in your environment)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment."

    try:
        # two common genai usage patterns: Client(...) or GenerativeModel(...)
        # 1) If genai has Client class -> use client.models.generate_content(...)
        if hasattr(genai, "Client"):
            client = genai.Client(api_key=api_key)
            # call the model; adapt model name if needed
            # `contents` can be a single string or a list of strings depending on library
            resp = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
        elif hasattr(genai, "GenerativeModel"):
            # older/alternate pattern
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-pro")
            resp = model.generate_content(prompt)
        else:
            return "Error: Unrecognized genai SDK shape (no Client or GenerativeModel)."

        if debug:
            print("--- DEBUG: raw response object ---")
            try:
                # attempt to pretty-print JSON if available
                print(json.dumps(resp.__dict__, default=str, indent=2))
            except Exception:
                print(repr(resp))
            print("---------------------------------")

        # Extract text from common response shapes:
        # - resp.text (simple)
        # - resp.output[0].content[0].text (newer structured output)
        # - resp.content (list)
        summary_text = None

        # 1) direct .text
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            summary_text = resp.text.strip()

        # 2) structured output: resp.output -> list -> dicts with 'content' that have 'text'
        if not summary_text:
            try:
                output = getattr(resp, "output", None)
                if output and isinstance(output, (list, tuple)) and len(output) > 0:
                    first = output[0]
                    # some SDKs put a `content` list inside the first output item
                    content = getattr(first, "content", None) or first.get("content") if isinstance(first, dict) else None
                    if content and isinstance(content, (list, tuple)) and len(content) > 0:
                        maybe_text = getattr(content[0], "text", None) or (content[0].get("text") if isinstance(content[0], dict) else None)
                        if maybe_text:
                            summary_text = maybe_text.strip()
            except Exception:
                # swallow; we'll try other shapes below
                pass

        # 3) resp.content (list of parts) common in some MCP responses
        if not summary_text and hasattr(resp, "content"):
            try:
                parts = resp.content
                if isinstance(parts, (list, tuple)) and parts:
                    # join all .text fields
                    texts = []
                    for p in parts:
                        t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
                        if t:
                            texts.append(t.strip())
                    if texts:
                        summary_text = "\n".join(texts)
            except Exception:
                pass

        # 4) fallback: attempt to string-cast resp
        if not summary_text:
            try:
                maybe = str(resp).strip()
                if maybe:
                    summary_text = maybe
            except Exception:
                summary_text = None

        if not summary_text:
            # Try to provide useful diagnostics if blocked or empty
            block_reason = None
            if hasattr(resp, "prompt_feedback") and getattr(resp, "prompt_feedback", None):
                pf = getattr(resp, "prompt_feedback")
                block_reason = getattr(pf, "block_reason", None)
            if block_reason:
                name = getattr(block_reason, "name", str(block_reason))
                return f"Summarization failed: request blocked by API. Reason: {name}"
            return "Summarization failed: API returned no usable text."

        return summary_text

    except Exception as e:
        # keep returns consistent with your original style
        return f"An error occurred during summarization: {type(e).__name__}: {e}"


def get_historical_data_for_ticker(ticker: str, days: int = 10) -> pd.DataFrame | None:
    """
    Fetch historical price data for a ticker over the past `days` days.
    Defaults to 10 days, but you can adjust.
    """
    try:
        yf_t = yf.Ticker(ticker)
        hist = yf_t.history(period=f"{days}d")
        if hist.empty:
            return None
        return hist
    except Exception as e:
        print(f"Failed to fetch historical data for {ticker}: {e}")
        return None


def build_context(ticker: str, news_summary: str) -> str:
    """
    Builds a context string that includes recent price movement + news summary.
    The LLM will see both historic trend + news.
    """
    hist = get_historical_data_for_ticker(ticker, days=10)
    if hist is not None:
        # compute 5-day trend
        recent = hist['Close'].pct_change().tail(5).sum()
        trend_desc = (
            f"{ticker} has moved {recent:.2%} over the past 5 trading days."
        )
    else:
        trend_desc = f"No historical price data for {ticker}."

    return f"{trend_desc}\n\nNews:\n{news_summary}"

def get_predictions_for_trending_tickers(debug:bool=False) -> dict:
    """Gets news and price predictions for a list of trending tickers. Use for broad queries like 'what are the predictions for trending stocks?'."""
    trending_tickers=get_trending_tickers()
    all_predictions = get_predictions_for_tickers(trending_tickers, debug=False)
    
    return all_predictions



def get_predictions_for_tickers(tickers: list[str], debug: bool = False) -> dict:
    """
    Gets news and generates price predictions for a list of tickers.

    Args:
        tickers: A list of stock ticker symbols.
        debug: If True, enables debug printing.

    Returns:
        A dictionary where keys are ticker symbols and values are the prediction strings.
    """
#print(f"--- Getting news for: {', '.join(tickers)} ---")
    original_summary = get_news_for_tickers(tickers)
    news_data = summarize_news_content(original_summary)

    
    predictions = {}
    
    print("\n--- Generating predictions ---")
    for ticker in tickers:
        # Check if we have news and at least one article with a summary
         # Call the single-prediction function
        prediction = predict_price_from_news(ticker, news_data, debug=debug)
        predictions[ticker] = prediction
    
            
    return predictions

if __name__ == "__main__":
       # original_summary = get_news_for_tickers(['ZYME','SOXS'])
     # 4. Summarize the original summary into key points
     #   meaningful_summary = summarize_news_content(original_summary)
     #   print("--- Meaningful Summary (from LLM) ---")
     #   print(f"{meaningful_summary}\n")

        # 5. Pass the original, more detailed summary to the prediction function
      #  prediction_result = predict_price_from_news('SOXS', original_summary)
      #  print("--- Price Prediction (from LLM) ---")
       # print(prediction_result)
      
        # Example: Get predictions for a list of tickers
        # sample_tickers = ['SOXL', 'GOOG', 'TSLA']
    
       # all_predictions = get_predictions_for_tickers(sample_tickers, debug=False)
    
        # print("\n\n--- Final Predictions ---")
       # for ticker, prediction in all_predictions.items():
          #  print(f"  - {ticker}: {prediction}")

        # print(get_final_predictions_for_tickers(debug=False))
        pass

