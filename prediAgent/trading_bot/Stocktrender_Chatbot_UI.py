import streamlit as st
import asyncio
import time
from datetime import datetime

# Add the parent directory of 'prediAgent' to the Python path
# to allow relative imports between sibling packages.
import sys
import os
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# --- Core Bot Imports ---
try:
    # Import the factory function to create agent components, not the components themselves.
    from Stocktrender_Chatbot_Agent_UI import create_agent_services, ADK_AVAILABLE, USER_ID
    if ADK_AVAILABLE:
        from google.genai import types
    AGENT_FACTORY_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import the chatbot agent factory. Make sure 'Stocktrender_Chatbot_Agent.py' is in the same directory. Error: {e}")
    AGENT_FACTORY_AVAILABLE = False
except Exception as e:
    st.error(f"An unexpected error occurred while initializing the agent factory: {e}")
    AGENT_FACTORY_AVAILABLE = False

# --- Streamlit Caching for Agent State ---
@st.cache_resource
def get_cached_agent_services():
    """
    Creates and caches the agent runner and session service.
    This function is decorated with @st.cache_resource, so it only runs ONCE.
    """
    if not AGENT_FACTORY_AVAILABLE:
        return None, None
    print("--- Creating and Caching Agent Services (should only see this once) ---")
    runner, session_service = create_agent_services()
    return runner, session_service

# Get the globally cached agent components.
runner, session_service = get_cached_agent_services()
AGENT_AVAILABLE = runner is not None and session_service is not None

# --- Event Loop Management ---
def get_or_create_event_loop():
    """Gets the current event loop or creates a new one if it doesn't exist."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:  # 'There is no current event loop...'
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Get the singleton event loop for this session.
loop = get_or_create_event_loop()

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="StockTrendAnalyzer", page_icon="📈")
st.title("📈 StockTrendAnalyzer Chatbot")
st.caption("Your AI-powered agent for stock market trends, predictions, and news.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with the stock market today?"}]

# --- Session Creation ---
# Ensure a session is created with the ADK's session service once per browser session.
async def create_session_async():
    """Asynchronously creates a new session and returns the session object."""
    try:
        # create_session() is async and requires app_name and user_id
        new_session_object = await session_service.create_session(
            app_name="StockTrendAnalyzer",
            user_id=USER_ID
        )
        return new_session_object
    except Exception as e:
        st.error(f"Failed to create a new session with the agent: {e}")
        return None

if "session_id" not in st.session_state and AGENT_AVAILABLE:
    # Use the persistent loop to run the async function
    session_object = loop.run_until_complete(create_session_async())
    if session_object:
        # Store only the ID of the session for the runner.
        st.session_state.session_id = session_object.id
        st.toast(f"New session created: {st.session_state.session_id}") # Optional: for debugging
    else:
        # If session creation fails, the agent is not available.
        AGENT_AVAILABLE = False

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---

# Accept user input
if prompt := st.chat_input("Ask about stocks, trends, or news..."):
    if not AGENT_AVAILABLE:
        st.error("The chatbot agent is not available. Please check the terminal for errors.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Define an async function to stream the response and return it
        async def stream_response():
            response_str = ""
            try:
                # Prepare the message for the ADK runner
                if ADK_AVAILABLE:
                    new_message = types.Content(role='user', parts=[types.Part(text=prompt)])
                else: # Fallback for mock runner
                    new_message = prompt
                
                # --- Call the ADK Runner ---
                response_generator = runner.run_async(
                    user_id=USER_ID, 
                    session_id=st.session_state.session_id, 
                    new_message=new_message
                )

                # Iterate through the async generator to get response chunks
                async for event in response_generator:
                    content = getattr(event, 'content', None)
                    parts = getattr(content, 'parts', []) if content else []
                    if parts:
                        text_chunk = getattr(parts[0], 'text', None)
                        if text_chunk:
                            response_str += text_chunk
                            message_placeholder.markdown(response_str + "▌")
                
                # Update the final message without the cursor
                message_placeholder.markdown(response_str)
                return response_str

            except Exception as e:
                error_message = f"An error occurred while processing your request: {e}"
                st.error(error_message)
                return "Sorry, I encountered an error. Please try again."

        full_response = ""
        try:
            # Use the persistent loop to run the async stream
            full_response = loop.run_until_complete(stream_response())
        except Exception as e:
            st.error(f"Error running async response handler: {e}")
            full_response = "Sorry, there was a problem running the stream."

    # Add the final assistant response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
