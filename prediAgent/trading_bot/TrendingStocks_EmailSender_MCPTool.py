import asyncio
import json
import os
import re
import sys
from typing import Dict, Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# This tool reuses the logic from the original mcp_trending_tckr_client.py

# optional: import genai if available in your environment
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()


def _fetch_tool_identifier_prompt() -> str:
    """Prompt that asks the model to return a STRICT JSON object."""
    return (
        "You have been given access to the below MCP Server Tools:\n\n"
        "{tools_description}\n\n"
        "You must identify the appropriate tool (name exactly as provided above) required to resolve the user query "
        "and return any arguments that should be passed to the tool. Output MUST be valid JSON, with these keys: "
        "\"user_query\", \"tool_identified\", \"arguments\" (where arguments is a JSON object or null).\n\n"
        "Do NOT include any extra text — only output the JSON.\n\n"
        "User Query:\n{user_query}\n\n"
    )


async def _generate_response_via_genai(user_query: str, tools_description: str) -> Optional[Dict[str, Any]]:
    """Try to use Gemini (genai) to produce the JSON."""
    if not GENAI_AVAILABLE: return None
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None

    try:
        client = genai.Client(api_key=api_key)
        prompt = _fetch_tool_identifier_prompt().format(user_query=user_query, tools_description=tools_description)
        response = client.models.generate_content(model="gemini-2.0-flash-001", contents=prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        first_json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if first_json_match:
            raw = first_json_match.group(0)
        data = json.loads(raw)
        return {
            "user_query": data.get("user_query"),
            "tool_identified": data.get("tool_identified") or data.get("tools_identified"),
            "arguments": data.get("arguments") if data.get("arguments") is not None else {}
        }
    except Exception as e:
        print(f"[mcp_tool_genai] Error: {e}")
        return None


def _generate_response_fallback(user_query: str, tools_description: str) -> Dict[str, Any]:
    """Rule-based fallback to identify the email tool."""
    tool_candidates = [name_match.group(1) for line in tools_description.splitlines() if (name_match := re.match(r"Tool\s*-\s*(\S+)\s*:", line))]
    email_tools = [t for t in tool_candidates if "email" in t.lower() or "send" in t.lower()]
    chosen_tool = email_tools[0] if email_tools else None
    
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_query)
    args = {"recipient_email": email_match.group(0)} if email_match else {}

    return {"user_query": user_query, "tool_identified": chosen_tool, "arguments": args}


async def _generate_response(user_query: str, tools_description: str) -> Dict[str, Any]:
    """Try genai -> fallback rules."""
    if GENAI_AVAILABLE:
        data = await _generate_response_via_genai(user_query, tools_description)
        if data: return data
    return _generate_response_fallback(user_query, tools_description)


async def run_mcp_email_tool(user_input: str):
    """Sends an email with stock information. Use when the user asks to 'send an email' or provides an email address."""
    print(f"[mcp_tool] Received query: {user_input}")
    
    server_script = "MCP_Trending_Tckr_EmailServer.py"
    server_cwd = "/Users/sangeeta/KaggleGoogleCapstoneProject/StockTrendAnalyzer/prediAgent/stock_Prediction_Mailer"
    server_path = os.path.join(server_cwd, server_script)

    print(f"[mcp_tool] DEBUG: Checking for server script at: {server_path}")
    if not os.path.exists(server_path):
        print(f"[mcp_tool] FATAL ERROR: Server script not found at '{server_path}'. Please check the path.")
        return

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        cwd=server_cwd,
    )

    print(f"[mcp_tool] DEBUG: Launching server with command: '{sys.executable} {server_script}' in directory '{server_cwd}'")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tools_description = "\n".join([f"Tool - {t.name}: {t.description}" for t in tools.tools])
                tool_names = [t.name for t in tools.tools]
                print(f"[mcp_tool] DEBUG: Found tools on server: {tool_names}")

                request_json = await _generate_response(user_query=user_input, tools_description=tools_description)
                tool_name = request_json.get("tool_identified")
                args = request_json.get("arguments") or {}
                print(f"[mcp_tool] DEBUG: Identified tool='{tool_name}' with args={args}")

                if not tool_name or tool_name not in tool_names:
                    print(f"[mcp_tool] ERROR: Could not identify a valid email tool from query. Available tools: {tool_names}")
                    return

                if not args.get("recipient_email"):
                    print("[mcp_tool] DEBUG: Recipient email not found in query, trying to get from environment variable 'DEFAULT_RECIPIENT_EMAIL'...")
                    args["recipient_email"] = os.getenv("DEFAULT_RECIPIENT_EMAIL", "")

                if not args.get("recipient_email"):
                    print("[mcp_tool] ERROR: No recipient email address found in query or environment variables. Cannot proceed.")
                    return

                print(f"[mcp_tool] Executing '{tool_name}' for recipient '{args.get('recipient_email')}'...")
                response = await session.call_tool(tool_name, arguments=args)
                
                print("[mcp_tool] DEBUG: Received response from server.")
                if response and hasattr(response, "content") and isinstance(response.content, list):
                    for item in response.content:
                        print(f"[mcp_tool_response] {getattr(item, 'text', item)}")
                else:
                    print(f"[mcp_tool_response] {response}")

    except Exception as e:
        import traceback
        print(f"[mcp_tool] FATAL ERROR: An exception occurred during execution.")
        traceback.print_exc()
