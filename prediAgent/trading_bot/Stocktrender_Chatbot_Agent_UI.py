"""chatbot_agent.py

Finalized script converted from the notebook. This version:
- Guards ADK imports and prints installation guidance if missing.
- If ADK is missing, provides a small mock runner/session so you can test the
    interactive REPL and verify responses without connecting to the real ADK.
- Robustly iterates and prints responses from Runner.run / run_async whether
    they are async generators, sync generators, awaitables, or plain values.

Usage:
    python chatbot_agent.py

Notes:
    - If you have the ADK installed and API keys configured, the script will
        use the real Runner and session service. Otherwise a mock will be used.
    - The interactive REPL accepts 'quit' or 'exit' to stop.
"""

from typing import Any, Dict, List, Union
import asyncio
import inspect
import sys
import os
import time
from datetime import datetime
import logging
import json
from logging.handlers import RotatingFileHandler

# Add the parent directory of 'prediAgent' to the Python path
# to allow relative imports between sibling packages.
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    # Import local trending tools if available
    from prediAgent.trading_bot.Stock_Trending_Tickers_Agent import (
        get_trending_tickers,
        get_predictions_for_trending_tickers,
        get_current_stock_price,
    )
    TRENDING_TOOLS_AVAILABLE = True
except Exception:
    TRENDING_TOOLS_AVAILABLE = False

try:
    # Import the A2A short-term predictor tool using its full path
    from prediAgent.short_Term_Stock_Predictor.A2AStockPredictionTool import get_a2a_short_term_prediction
    A2A_FORCASTER_AVAILABLE = True
except Exception:
    A2A_FORCASTER_AVAILABLE = False

try:
    # Import the MCP email tool
    from prediAgent.trading_bot.TrendingStocks_EmailSender_MCPTool import run_mcp_email_tool
    MCP_EMAIL_TOOL_AVAILABLE = True
except Exception:
    MCP_EMAIL_TOOL_AVAILABLE = False

try:
    # Import news predictor tool
    from prediAgent.trading_bot.Stock_Predictor_FromNews_Agent import predict_ticker_from_news
    NEWS_PREDICTOR_AVAILABLE = True
except Exception:
    # Avoid eager import-time failures and ensure the module is imported lazily
    # inside the A@A handler. Initialize safe defaults so the handler uses
    # globals() to populate these when/if the module is imported later.
    NEWS_PREDICTOR_AVAILABLE = False
    predict_ticker_from_news = None


# -----------------------------
# Safe, guarded ADK imports
# -----------------------------
ADK_AVAILABLE = False
try:
    # Try importing ADK symbols used in the original notebook
    from google.adk.agents import Agent, LlmAgent  # type: ignore
    from google.adk.apps.app import App, EventsCompactionConfig  # type: ignore
    from google.adk.models.google_llm import Gemini  # type: ignore
    from google.adk.sessions import DatabaseSessionService, InMemorySessionService  # type: ignore
    from google.adk.runners import Runner  # type: ignore
    from google.adk.tools.tool_context import ToolContext  # type: ignore
    from google.genai import types  # type: ignore

    # Optional convenience import
    from google import genai  # type: ignore

    ADK_AVAILABLE = True
    print("✅ ADK components imported successfully.")
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)
    print("ADK packages are not installed in this environment. Using a mock runner so you can test the REPL.")
except Exception as e:
    print("Unexpected import error:", e)


# --- latency logger -------------------------------------------------------
LATENCY_LOGFILE = os.path.join(os.path.dirname(__file__), 'Stocktrender_Chatbot_Agent_Log.log')
lat_logger = logging.getLogger('stocktrender_latency')
if not lat_logger.handlers:
    lat_logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LATENCY_LOGFILE)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt)
    lat_logger.addHandler(fh)

# Truncate existing latency log for a fresh run. This ensures each invocation
# of the script starts with a clean file and a short timestamped header.
try:
    # If file exists, overwrite it with a small header; otherwise create it.
    if os.path.exists(LATENCY_LOGFILE):
        with open(LATENCY_LOGFILE, "w", encoding="utf-8") as _lf:
            _lf.write(f"# stocktrender_chatbot.log\n# Truncated at: {datetime.utcnow().isoformat()}Z\n\n")
    else:
        # ensure the file is created
        open(LATENCY_LOGFILE, "w", encoding="utf-8").close()
    print(f"Cleared latency log: {LATENCY_LOGFILE}")
except Exception as _e:
    print(f"Warning: unable to clear latency logfile {LATENCY_LOGFILE}: {_e}")

# -----------------------------
# Structured JSONL trace logger (optional)
# -----------------------------
TRACE_LOGFILE = os.path.join(os.path.dirname(__file__), 'Stocktrender_Chatbot_Agent_Trace.log')

class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
        }
        try:
            if isinstance(record.msg, dict):
                payload.update(record.msg)
            else:
                payload["message"] = record.getMessage()
        except Exception:
            payload["message"] = record.getMessage()
        if record.exc_info:
            try:
                payload["exc"] = self.formatException(record.exc_info)
            except Exception:
                payload["exc"] = repr(record.exc_info)
        return json.dumps(payload, default=str)

TRACE_ENABLED = os.environ.get("TRACE_LOG", "").lower() not in ("", "0", "false", "no")
trace_logger = logging.getLogger('stocktrender_trace')
if TRACE_ENABLED and not trace_logger.handlers:
    trace_logger.setLevel(logging.INFO)
    _thr = RotatingFileHandler(TRACE_LOGFILE, maxBytes=5_000_000, backupCount=5)
    _thr.setLevel(logging.INFO)
    _thr.setFormatter(JsonLineFormatter())
    trace_logger.addHandler(_thr)
    trace_logger.propagate = False
    trace_logger.info({"event": "trace.enabled"})


# -----------------------------
# Configuration
# -----------------------------
APP_NAME = "StockTrendAnalyzer"
USER_ID = "default"
MODEL_NAME = "gemini-2.5-flash-lite"

# Retry config: prefer ADK types when available, else a plain dict
try:
    if ADK_AVAILABLE and 'types' in globals():
        retry_config = types.HttpRetryOptions(
            attempts=5,
            exp_base=7,
            initial_delay=1,
            http_status_codes=[429, 500, 503, 504],
        )
    else:
        retry_config = dict(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])
except Exception:
    retry_config = dict(attempts=3, initial_delay=1)


# -----------------------------
# If ADK not available, provide simple mocks so you can test the REPL locally.
# -----------------------------
class MockPart:
    def __init__(self, text: str):
        self.text = text


class MockContent:
    def __init__(self, text: str):
        self.parts = [MockPart(text)]


class MockEvent:
    def __init__(self, text: str):
        self.content = MockContent(text)


import uuid

class MockSessionService:
    def __init__(self):
        self._sessions = {}  # Use a dict to store mock session objects

    async def create_session(self, app_name: str, user_id: str):
        """Creates a new session and returns a mock session object with an ID."""
        session_id = str(uuid.uuid4())
        # The real service returns a Session object, so we mock that.
        mock_session = type('Session', (), {'id': session_id})()
        self._sessions[session_id] = mock_session
        return mock_session

    async def get_session(self, app_name: str, user_id: str, session_id: str):
        """Gets an existing session or returns None."""
        return self._sessions.get(session_id)


class MockRunner:
    """Simple mock runner that returns canned responses for testing the REPL.

    It provides both a sync .run (returns list) and an async generator .run_async
    to mimic different ADK behaviors.
    """

    async def run_async(self, user_id: str, session_id: str, new_message: Union[str, Any]):
        # Async generator: yield a single MockEvent after a short delay
        await asyncio.sleep(0.05)
        if isinstance(new_message, str):
            text = f"Echo (sim): {new_message}"
        else:
            text = "Echo (sim): <complex message>"
        yield MockEvent(text)

    def run(self, user_id: str, session_id: str, new_message: Union[str, Any]):
        # Sync path: return a list of events
        if isinstance(new_message, str):
            return [MockEvent(f"Echo (sim): {new_message}")]
        return [MockEvent("Echo (sim): <complex message>")]


def create_agent_services():
    """Creates and returns the ADK runner and session_service."""
    # Choose real or mock services
    if ADK_AVAILABLE:
        # Only attempt to use the real ADK if we have credentials configured.
        has_api_key = bool(os.environ.get('GOOGLE_API_KEY'))
        has_vertex_creds = bool(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')) and bool(os.environ.get('GOOGLE_CLOUD_PROJECT')) and bool(os.environ.get('GOOGLE_CLOUD_LOCATION'))
        if not (has_api_key or has_vertex_creds):
            print("ADK is installed but no API credentials were detected.")
            print("Set GOOGLE_API_KEY for the Google GenAI API or set GOOGLE_APPLICATION_CREDENTIALS + GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION for Vertex AI.")
            print("Falling back to mock runner so you can test the REPL locally.")
            local_session_service = MockSessionService()
            local_runner = MockRunner()
        else:
            # Try to create real ADK services
            try:
                # Use a database-backed session service to persist state across reruns.
                # Put the database in a .data directory at the project root to avoid CWD/permission issues.
                data_dir = os.path.join(_PARENT_DIR, ".data")
                os.makedirs(data_dir, exist_ok=True) # Ensure the directory exists
                db_path = os.path.join(data_dir, "streamlit_sessions.db")
                db_url = f"sqlite+aiosqlite:///{db_path}"
                print(f"✅ Using database at: {db_path}")
                local_session_service = DatabaseSessionService(db_url=db_url)

                # --- ADK Tool Registration ---
                adk_tools = []
                if TRENDING_TOOLS_AVAILABLE:
                    adk_tools.extend([
                        get_trending_tickers,
                        get_predictions_for_trending_tickers,
                        get_current_stock_price,
                    ])
                if A2A_FORCASTER_AVAILABLE:
                    adk_tools.append(get_a2a_short_term_prediction)
                if MCP_EMAIL_TOOL_AVAILABLE:
                    adk_tools.append(run_mcp_email_tool)
                if NEWS_PREDICTOR_AVAILABLE and predict_ticker_from_news is not None:
                    adk_tools.append(predict_ticker_from_news)
                
                print(f"✅ Registering {len(adk_tools)} functions as tools with the ADK Agent.")

                root_agent = Agent(
                    model=Gemini(model=MODEL_NAME, retry_options=retry_config),
                    tools=adk_tools,
                    name="stock_trend_analyzer_agent",
                    description="A chatbot that can provide stock market trends, predictions, and send email summaries."
                )

                local_runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=local_session_service)
                print("✅ Stateful agent initialized (real ADK).")
                try:
                    trace_logger.info({"event": "runner.init", "adk": True, "app": APP_NAME})
                except Exception:
                    pass
            except Exception as e:
                print('Failed to create real ADK runner/session:', e)
                print('Falling back to mock runner.')
                try:
                    trace_logger.exception({"event": "runner.init.failed", "err": str(e)})
                except Exception:
                    pass
                local_session_service = MockSessionService()
                local_runner = MockRunner()
    else:
        local_session_service = MockSessionService()
        local_runner = MockRunner()
        print("Using mock runner/session_service. You can test the REPL without ADK.")
        try:
            trace_logger.info({"event": "runner.init.mock", "app": APP_NAME})
        except Exception:
            pass
    return local_runner, local_session_service

# Create global instances for the interactive REPL (if run directly)
# The Streamlit UI will call create_agent_services() within a cached function.
runner, session_service = create_agent_services()



# Simple in-memory prediction store so a prediction can be started and retrieved later.
# Keys: upper-case ticker symbol -> dict(status: 'running'|'done'|'error', result, started)
# (Prediction store and background executor removed to preserve original behavior.)


# -----------------------------
# Helper to consume and print responses from runners robustly
# -----------------------------
async def _consume_and_print(result):
    """Given a result which may be an async generator, awaitable, generator, iterable
    or plain value, consume and print textual content found in event.content.parts.
    It also logs any tool calls it encounters.
    """
    # If it's an awaitable (but not yet awaited), await it
    if inspect.isawaitable(result):
        result = await result

    async def process_event(event):
        if isinstance(event, (str, bytes)):
            print(event)
            return
        
        content = getattr(event, 'content', None)
        if not content or not hasattr(content, 'parts'):
            return

        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                print(part.text)
            elif hasattr(part, 'function_call'):
                # Log the tool call for debugging purposes
                fc = part.function_call
                print(f"[AGENT] Calling tool: {fc.name}({fc.args})")
                try:
                    trace_logger.info({"event": "tool.call", "name": fc.name, "args": fc.args})
                except Exception:
                    pass # Ignore logging errors

    # Async generator?
    if hasattr(result, '__aiter__'):
        async for event in result:
            await process_event(event)
        return

    # If it's a sync generator or iterable (but not str/bytes)
    if inspect.isgenerator(result) or (hasattr(result, '__iter__') and not isinstance(result, (str, bytes))):
        for event in result:
            # Since process_event is async, we need a running loop.
            # This path is less common with the ADK runner.
            await process_event(event)
        return

    # Plain value: process the single event
    await process_event(result)


async def run_session(runner_instance: Any = None, user_queries: Union[List[str], str, None] = None, session_name: str = 'default'):
    """Create/get session and send queries using runner_instance. Prints responses.

    This function handles multiple runner shapes (async generator, sync generator, awaitable, list, or single value).
    """
    if runner_instance is None:
        print('No runner instance provided.')
        return

    try:
        trace_logger.info({"event": "session.run.start", "session": session_name, "queries": user_queries})
    except Exception:
        pass

    print(f"\n ### Session: {session_name}")

    # Create/get session: Get the session if it exists, otherwise create it.
    session = None
    try:
        # First, try to get the session
        if hasattr(session_service, 'get_session'):
            get_attr = getattr(session_service, 'get_session')
            try:
                trace_logger.info({"event": "session.get.attempt", "session": session_name, "app": APP_NAME})
            except Exception:
                pass
            if inspect.iscoroutinefunction(get_attr):
                session = await get_attr(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)
            else:
                session = get_attr(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)
        
        # If get_session returned None or didn't exist, try creating it
        if session is None and hasattr(session_service, 'create_session'):
            create_attr = getattr(session_service, 'create_session')
            try:
                trace_logger.info({"event": "session.create.attempt", "session": session_name, "app": APP_NAME})
            except Exception:
                pass
            if inspect.iscoroutinefunction(create_attr):
                session = await create_attr(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)
            else:
                session = create_attr(app_name=APP_NAME, user_id=USER_ID, session_id=session_name)

        # If both failed, use a dummy session
        if session is None:
             session = type('S', (), {'id': session_name})()

    except Exception as e:
        print('Warning: session_service create/get failed; falling back to dummy session. Error:', e)
        try:
            trace_logger.exception({"event": "session.createget.failed", "session": session_name, "err": str(e)})
        except Exception:
            pass
        session = type('S', (), {'id': session_name})()


    sess_id = getattr(session, 'id', session_name)

    if not user_queries:
        print('No queries!')
        return

    if isinstance(user_queries, str):
        user_queries = [user_queries]

    for query_text in user_queries:
        try:
            trace_logger.info({"event": "query.received", "session": sess_id, "query": query_text})
        except Exception:
            pass
        # The ADK agent now handles all tool calls through natural language.
        # The manual `_handle_local_tools` function has been removed to allow
        # the agent to correctly dispatch to the registered tools.

        # Build message: prefer ADK types when available
        if ADK_AVAILABLE and 'types' in globals():
            try:
                new_message = types.Content(role='user', parts=[types.Part(text=query_text)])
            except Exception:
                new_message = query_text
        else:
            new_message = query_text

        # Try run_async first
        if hasattr(runner_instance, 'run_async'):
            run_async_attr = getattr(runner_instance, 'run_async')
            try:
                start_t = time.monotonic()
                try:
                    trace_logger.info({"event": "runner.call.start", "session": sess_id, "query": query_text, "method": "run_async"})
                except Exception:
                    pass
                if inspect.iscoroutinefunction(run_async_attr):
                    result = run_async_attr(user_id=USER_ID, session_id=sess_id, new_message=new_message)
                    await _consume_and_print(result)
                else:
                    # sync function: may return iterable/awaitable
                    result = run_async_attr(user_id=USER_ID, session_id=sess_id, new_message=new_message)
                    await _consume_and_print(result)
                elapsed = time.monotonic() - start_t
                lat_logger.info(f"runner_call=run_async session={sess_id} query={query_text!r} elapsed_s={elapsed:.3f}")
                try:
                    trace_logger.info({"event": "runner.call.end", "session": sess_id, "elapsed_ms": int(elapsed*1000)})
                except Exception:
                    pass
                continue
            except Exception as e:
                try:
                    trace_logger.exception({"event": "runner.call.error", "method": "run_async", "err": str(e)})
                except Exception:
                    pass
                print('run_async path failed:', e)

        # Fallback to run
        if hasattr(runner_instance, 'run'):
            run_attr = getattr(runner_instance, 'run')
            try:
                start_t = time.monotonic()
                try:
                    trace_logger.info({"event": "runner.call.start", "session": sess_id, "query": query_text, "method": "run"})
                except Exception:
                    pass
                if inspect.iscoroutinefunction(run_attr):
                    result = await run_attr(user_id=USER_ID, session_id=sess_id, new_message=new_message)
                    await _consume_and_print(result)
                else:
                    result = run_attr(user_id=USER_ID, session_id=sess_id, new_message=new_message)
                    await _consume_and_print(result)
                elapsed = time.monotonic() - start_t
                lat_logger.info(f"runner_call=run(session-async if coroutine else sync) session={sess_id} query={query_text!r} elapsed_s={elapsed:.3f}")
                try:
                    trace_logger.info({"event": "runner.call.end", "session": sess_id, "elapsed_ms": int(elapsed*1000)})
                except Exception:
                    pass
            except Exception as e:
                try:
                    trace_logger.exception({"event": "runner.call.error", "method": "run", "err": str(e)})
                except Exception:
                    pass
                print('run path failed:', e)
        else:
            print('Runner does not support run_async or run; cannot stream responses.')


async def interactive_async(session_name: str = "stateful-agentic-session"):
    """Start an interactive prompt. Each user input is sent to run_session and responses are printed."""
    print('\nInteractive chat started. Type your messages below (quit/exit to stop).')
    print('You can now ask for predictions, trends, or emails in natural language.')

    loop = asyncio.get_event_loop()
    while True:
        try:
            # Short, clear prompt. `input()` text kept concise to avoid wrapping in terminals.
            user_input = await loop.run_in_executor(None, lambda: input('StockTrendAnalyzer — enter query (type "help" for commands): ').strip())
        except (EOFError, KeyboardInterrupt):
            print('\nExiting interactive chat.')
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit'):
            print('Goodbye.')
            break

        try:
            await run_session(runner, user_queries=[user_input], session_name=session_name)
        except Exception as e:
            print('Error during run_session:', e)

    print('Interactive session ended.')


def main():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        print('Detected running event loop; scheduling interactive chat.')
        asyncio.ensure_future(interactive_async())
    else:
        asyncio.run(interactive_async())


if __name__ == '__main__':
    main()