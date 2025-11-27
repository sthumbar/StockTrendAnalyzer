import asyncio
import json
import os
import re
from typing import Dict, Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# optional: import genai if available in your environment
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


def fetch_tool_identifier_prompt() -> str:
    """
    Prompt that asks the model to return a STRICT JSON object.
    Keys must be quoted and values must be valid JSON.
    """
    return (
        "You have been given access to the below MCP Server Tools:\n\n"
        "{tools_description}\n\n"
        "You must identify the appropriate tool (name exactly as provided above) required to resolve the user query "
        "and return any arguments that should be passed to the tool. Output MUST be valid JSON, with these keys: "
        "\"user_query\", \"tool_identified\", \"arguments\" (where arguments is a JSON object or null).\n\n"
        "Do NOT include any extra text — only output the JSON.\n\n"
        "User Query:\n{user_query}\n\n"
        "Example output:\n"
        '{{\n'
        '  "user_query": "What are trending tickers?",\n'
        '  "tool_identified": "get_trending_tickers",\n'
        '  "arguments": {{"limit": 10}}\n'
        '}}\n'
    )


async def generate_response_via_genai(user_query: str, tools_description: str) -> Optional[Dict[str, Any]]:
    """
    Try to use Gemini (genai) to produce the JSON. Returns dict on success or None on failure.
    """
    if not GENAI_AVAILABLE:
        return None

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[genai] GEMINI_API_KEY not set in environment. Skipping GenAI.")
        return None

    try:
        client = genai.Client(api_key=api_key)
        prompt = fetch_tool_identifier_prompt().format(user_query=user_query, tools_description=tools_description)

        # use the recommended API method for text generation where available
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt
        )

        raw = response.text.strip()
        # strip code fences and try to load JSON
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Some models return extra text — extract first {...} block
        first_json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if first_json_match:
            raw = first_json_match.group(0)

        data = json.loads(raw)
        # Normalize keys
        data = {
            "user_query": data.get("user_query"),
            "tool_identified": data.get("tool_identified") or data.get("tools_identified"),
            "arguments": data.get("arguments") if data.get("arguments") is not None else {}
        }
        # If arguments are a string like "limit:10" or "limit,10", try to parse
        if isinstance(data["arguments"], str):
            # try key:value pairs comma separated
            args = {}
            for part in re.split(r"[;,]+", data["arguments"]):
                if ":" in part:
                    k, v = part.split(":", 1)
                    k = k.strip().strip('"').strip("'")
                    v = v.strip()
                    # try to interpret numbers/bools
                    if re.fullmatch(r"\d+", v):
                        v_parsed = int(v)
                    else:
                        try:
                            v_parsed = json.loads(v)
                        except Exception:
                            v_parsed = v.strip('"').strip("'")
                    args[k] = v_parsed
                elif part.strip():
                    # single token, treat as flag True
                    args[part.strip()] = True
            data["arguments"] = args
        elif data["arguments"] is None:
            data["arguments"] = {}

        return data

    except Exception as e:
        print(f"[genai] Error calling genai: {e}")
        return None


def generate_response_fallback(user_query: str, tools_description: str) -> Dict[str, Any]:
    """
    Rule-based fallback: choose an email tool if the query asks to email/send,
    otherwise choose a ticker/trending tool. Also extract simple args from text.
    """
    query_lower = user_query.lower()

    # Heuristics
    wants_email = any(token in query_lower for token in ("email", "e-mail", "mail", "send"))
    wants_tickers = any(token in query_lower for token in ("ticker", "tickers", "trending", "top", "trend"))

    # Parse optional numeric limit
    limit_match = re.search(r"\b(top|last|limit)\s*(?:=|\:)?\s*(\d{1,3})\b", query_lower)
    limit = int(limit_match.group(2)) if limit_match else None

    # Parse email address if present
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_query)
    recipient_email = email_match.group(0) if email_match else None

    # Parse format html/plain
    wants_html = bool(re.search(r"\bhtml\b", query_lower))

    # Scan tool names from description
    tool_candidates = []
    for line in tools_description.splitlines():
        name_match = re.match(r"Tool\s*-\s*(\S+)\s*:", line)
        if name_match:
            tool_candidates.append(name_match.group(1))

    # Choose tool
    chosen_tool = None
    if wants_email:
        email_tools = [t for t in tool_candidates if ("email" in t.lower() or "send" in t.lower())]
        if email_tools:
            chosen_tool = email_tools[0]
    if not chosen_tool:
        # prefer ticker/trending tools
        for t in tool_candidates:
            if "ticker" in t.lower() or "tick" in t.lower() or "trend" in t.lower():
                chosen_tool = t
                break

    # Build args
    args: Dict[str, Any] = {}
    if limit:
        args["limit"] = limit
    if recipient_email:
        args["recipient_email"] = recipient_email
    if wants_html:
        args["format"] = "html"

    return {
        "user_query": user_query,
        "tool_identified": chosen_tool,
        "arguments": args,
    }


async def generate_response(user_query: str, tools_description: str) -> Dict[str, Any]:
    """
    Try genai -> fallback rules. Always returns a dict with keys:
    user_query, tool_identified (may be None), arguments (dict).
    """
    # Try GenAI if available
    if GENAI_AVAILABLE:
        data = await generate_response_via_genai(user_query, tools_description)
        if data:
            return data

    # fallback
    return generate_response_fallback(user_query, tools_description)


async def main(user_input: str):
    print("-" * 60)
    print("User Input:", user_input)

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_trending_tckr_server.py"],  # your MCP server script
        cwd="/Users/sangeeta/Documents/kaggleHakathonNov15"  # update to your working directory
    )

    try:
        async with stdio_client(server_params) as (read, write):
            print("[mcp] Connection established, creating session...")
            try:
                async with ClientSession(read, write) as session:
                    print("[mcp] Session created, initializing...")
                    try:
                        await session.initialize()
                        print("[mcp] MCP session initialized")

                        tools = await session.list_tools()
                        tools_description = ""
                        tool_names = []
                        for each_tool in tools.tools:
                            name = each_tool.name
                            desc = each_tool.description or ""
                            tool_names.append(name)
                            tools_description += f"Tool - {name}:\n{desc}\n\n"

                        print("[mcp] Available tools:", tool_names)

                        request_json = await generate_response(user_query=user_input, tools_description=tools_description)

                        tool_name = request_json.get("tool_identified")
                        args = request_json.get("arguments") or {}

                        # Ensure required arguments for specific tools are supplied
                        if tool_name == "send_tickers_by_email":
                            # Try to recover missing recipient_email interactively or from env
                            if not args.get("recipient_email"):
                                default_rcpt = os.getenv("DEFAULT_RECIPIENT_EMAIL", "").strip()
                                prompt = "Enter recipient_email (press Enter to use DEFAULT_RECIPIENT_EMAIL): " if default_rcpt else "Enter recipient_email: "
                                user_rcpt = input(prompt).strip()
                                args["recipient_email"] = user_rcpt or default_rcpt

                            # Coerce limit to int if provided as string
                            if "limit" in args and isinstance(args["limit"], str):
                                try:
                                    args["limit"] = int(args["limit"])  # best-effort conversion
                                except Exception:
                                    pass

                            # Normalize format
                            if "format" in args and isinstance(args["format"], str):
                                fmt = args["format"].lower().strip()
                                if fmt not in ("text", "html"):
                                    fmt = "text"
                                args["format"] = fmt

                        if not tool_name:
                            print("[client] No suitable tool identified by AI/fallback.")
                            print("-" * 60)
                            return

                        if tool_name not in tool_names:
                            print(f"[client] Identified tool '{tool_name}' is NOT present in server tools. Available: {tool_names}")
                            print("[client] Aborting tool call.")
                            print("-" * 60)
                            return

                        print(f"[client] Executing tool '{tool_name}' with arguments: {args}")
                        response = await session.call_tool(tool_name, arguments=args)

                        # response.content may be list of content items; guard accordingly
                        if response is None:
                            print("[client] No response returned from session.call_tool()")
                        else:
                            try:
                                # Try to print readable text result(s)
                                if hasattr(response, "content") and isinstance(response.content, list):
                                    for item in response.content:
                                        # Some items may have .text or .json
                                        if hasattr(item, "text"):
                                            print(item.text)
                                        else:
                                            # fallback to raw repr
                                            print(item)
                                else:
                                    print(response)
                            except Exception:
                                print("Response (raw):", response)

                        print("-" * 60)
                    except Exception as e:
                        print(f"[mcp] Session initialization / tool call error: {e}")
            except Exception as e:
                print(f"[mcp] Session creation error: {e}")
    except Exception as e:
        print(f"[mcp] Connection error: {e}")


if __name__ == "__main__":
    try:
        while True:
            query = input("What is your query? → ").strip()
            if not query:
                continue
            # run the asyncio main per-query
            asyncio.run(main(query))
    except KeyboardInterrupt:
        print("\nExiting.")
