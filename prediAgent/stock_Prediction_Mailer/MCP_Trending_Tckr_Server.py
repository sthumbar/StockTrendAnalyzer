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
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("Stock Server")




def get_trending_tickers_without_selenium():
    url = "https://www.stockmarketwatch.com/markets/pre-market/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        print("came here") 
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
                    print("Volume Text:", volume_text)  # For debugging
                    try:
                        numeric_volume = 0
                        if 'M' in volume_text:
                            numeric_volume = float(volume_text.replace('M', '')) * 1_000_000
                        elif 'K' in volume_text:
                            numeric_volume = float(volume_text.replace('K', '')) * 1_000
                        else:
                            numeric_volume = float(volume_text)

                        print("Parsed Volume:", numeric_volume)  # For debugging
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
    
    
def get_trending_tickers_without_selenium_old():
    url = "https://www.stockmarketwatch.com/markets/pre-market/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        print("came here") 
        # Find the table containing the tickers
        table = soup.find("table", {"class": "tbldata"})  # Replace "datatable" with the actual class name
        
        # Extract tickers from the table rows
        tickers = []
        if table:
            rows = table.find_all("tr")[1:]  # Skip the header row
            for row in rows:
              
                columns = row.find_all("td")
                if columns:
                    print("Sangeeta *** ",columns[4].text.strip() )  # For debugging
                    ticker = columns[2].text.strip()  # Assuming the first column contains the ticker symbol
                    tickers.append(ticker)
        
        return tickers
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return []
    

@mcp.tool()
def mcp_extract_tickers():
    """
    MCP Tool: Extract trending tickers using the specified method.

 """
    
    return get_trending_tickers_without_selenium()

def _open_smtp(smtp_server: str, smtp_port: int, use_ssl: bool | None = None):
    """Create an SMTP client with SSL or STARTTLS based on port or explicit flag."""
    if use_ssl is None:
        use_ssl = smtp_port == 465
    if use_ssl:
        return smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
    client = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
    client.starttls()
    return client


@mcp.tool()
def send_tickers_by_email(recipient_email: str, limit: int = 25, format: str = "text", subject: str | None = None, use_ssl: bool | None = None) -> str:
    """Send trending tickers to an email address.

    Args:
        recipient_email: Destination email address.
        limit: Maximum number of tickers to include (default 25).
        format: "text" or "html" body format.

    Env Requirements:
        SENDER_EMAIL, SENDER_PASSWORD (or APP PASSWORD), SMTP_SERVER, SMTP_PORT

    Returns:
        Status message string.
    """
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")

    if not all([sender_email, sender_password, smtp_server, smtp_port]):
        return (
            "Missing email configuration. Set SENDER_EMAIL, SENDER_PASSWORD, SMTP_SERVER, SMTP_PORT in environment/.env"
        )

    try:
        smtp_port = int(smtp_port)
    except (ValueError, TypeError):
        return "SMTP_PORT must be an integer."

    tickers = get_trending_tickers_without_selenium()
    if not tickers:
        return "No tickers retrieved to send."

    tickers = tickers[: max(1, limit)]

    subject = subject or "Trending Stock Tickers"
    if format.lower() == "html":
        body_content = "".join(f"<li>{t}</li>" for t in tickers)
        body = f"<html><body><h3>Trending Tickers</h3><ul>{body_content}</ul></body></html>"
        msg = MIMEText(body, "html")
    else:
        body = "Trending Tickers:\n" + "\n".join(tickers)
        msg = MIMEText(body)

    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    print("[EMAIL LOG] Attempting to send email...")
    print(f"[EMAIL LOG] From: {sender_email}, To: {recipient_email}, Server: {smtp_server}:{smtp_port}")

    try:
        with _open_smtp(smtp_server, smtp_port, use_ssl) as server:
            print("[EMAIL LOG] SMTP connection successful.")
            server.login(sender_email, sender_password)
            print("[EMAIL LOG] SMTP login successful.")
            server.send_message(msg)
            print("[EMAIL LOG] Email sent successfully via send_message.")
        return f"Sent {len(tickers)} tickers to {recipient_email}."
    except SMTPAuthenticationError as e:
        error_message = f"Email send failed: authentication error ({e.smtp_code} {e.smtp_error}). Check SENDER_EMAIL/SENDER_PASSWORD (use app password if needed)."
        print(f"[EMAIL LOG] {error_message}")
        return error_message
    except (SMTPConnectError, ConnectionError, TimeoutError) as e:
        error_message = f"Email send failed: connection error ({e}). Verify SMTP_SERVER/SMTP_PORT and network."
        print(f"[EMAIL LOG] {error_message}")
        return error_message
    except SMTPException as e:
        error_message = f"Email send failed: SMTP error ({e})."
        print(f"[EMAIL LOG] {error_message}")
        return error_message
    except Exception as e:
        error_message = f"Email send failed: {e}"
        print(f"[EMAIL LOG] {error_message}")
        return error_message

@mcp.tool()
def test_email_config(use_ssl: bool | None = None) -> str:
    """Diagnose email configuration without sending a message.

    Attempts to load env vars and log in to the SMTP server.
    Returns a human-readable diagnostic string.
    """
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")

    missing = [k for k, v in {
        "SENDER_EMAIL": sender_email,
        "SENDER_PASSWORD": sender_password,
        "SMTP_SERVER": smtp_server,
        "SMTP_PORT": smtp_port,
    }.items() if not v]
    if missing:
        return "Missing env vars: " + ", ".join(missing)

    try:
        smtp_port_int = int(smtp_port)
    except (ValueError, TypeError):
        return "SMTP_PORT must be an integer."

    try:
        with _open_smtp(smtp_server, smtp_port_int, use_ssl) as server:
            # Some providers require EHLO; smtplib does this implicitly
            server.login(sender_email, sender_password)
        mode = "SSL" if (use_ssl or smtp_port_int == 465) else "STARTTLS"
        return f"SMTP login succeeded using {mode} on {smtp_server}:{smtp_port_int} as {sender_email}."
    except SMTPAuthenticationError as e:
        return f"SMTP auth failed ({e.smtp_code} {e.smtp_error}). Ensure correct credentials/app password."
    except (SMTPConnectError, ConnectionError, TimeoutError) as e:
        return f"SMTP connection failed: {e}. Verify server/port and network access."
    except SMTPException as e:
        return f"SMTP error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

@mcp.tool()
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
            if news:
                all_news.append(f"--- News for {ticker_symbol.upper()} ---")
                # Limit to the top 5 news articles for brevity
                for item in news[:5]:
                    all_news.append(f"  - Title: {item['title']}")
                    all_news.append(f"    Link: {item['link']}")
                all_news.append("") # Add a blank line for readability
        except Exception as e:
            all_news.append(f"Could not fetch news for {ticker_symbol}: {e}")

    if not all_news:
        return "No news found for the provided tickers."

    return "\n".join(all_news)


if __name__ == "__main__":
    # Start FastMCP server (single invocation).
    mcp.run()


