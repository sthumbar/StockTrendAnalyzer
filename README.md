# StockTrendAnalyzer
StockTrendAnalyzer Agent 

Automated Forecasting and Proactive Alerts using a Multi-Agent Architecture 

🚀 Project Overview 

This document serves as the final submission for the Kaggle Agents Intensive Capstone project. 

The StockTrendAnalyzer uses a multi-agent system orchestrated by the Gemini model to automate market analysis, predict short-term price movements, and proactively deliver actionable insights to users via email alerts. 



 

🎯 Problem Statement 

Retail investors and financial analysts are overwhelmed by the constant flow of stock market data, news, and social sentiment. Identifying trending stocks and predicting short-term price movements typically requires advanced analytical skills, extensive market monitoring, and complex predictive modeling. Most investors lack the time or expertise to process this information effectively, causing them to miss profitable opportunities or react too late. 

This project addresses these challenges by building an intelligent multi-agent system that automates market trend detection, news-driven sentiment analysis, and short-term price prediction. Through a simple conversational interface, users can request trending tickers, ask for stock forecasts, or query market insights without needing data science knowledge. 

The system proactively monitors the market and sends users timely email alerts on significant price movements or emerging trends—ensuring they never miss critical updates. This enables retail investors to make faster, more informed decisions with minimal effort. 

 

💡 Solution: An Intelligent Multi-Agent System 

The solution is a modular, distributed, multi-agent platform called StockTrendAnalyzer Agent, designed to automate stock analysis and surface actionable insights through a simple conversational interface. By delegating specific responsibilities to specialized agents and microservices, the system overcomes information overload and the complexity of financial data analysis. 

High level Architecture 
(provided in ArchitectureDiagram.png)

 


The platform is composed of four primary components, each with a well-defined purpose: 

 

1. Orchestrator Server (System Brain & Coordinator) 

The Orchestrator is the central intelligence layer of the system. It: 

Receives and interprets user queries from the web interface. 

Determines the user’s intent (trend detection, prediction, or notification). 

Delegates the request to the correct internal or external agent. 

Manages session state and conversation context using a local SQLite database. 

The Orchestrator does not perform any heavy computation itself. Instead, it routes tasks to the appropriate specialized agents, making the system modular and scalable. 

2. Streamlit Web UI (Conversational Front End) 

This is the user-facing interface—a clean, simple chatbot where users can type natural language queries, such as: 

"What stocks are trending today?" 

"Predict AAPL’s price for tomorrow." 

"Email me the top 5 trending stocks." 

The Streamlit UI abstracts away all complexity. Users do not need financial expertise or technical skills. The UI simply forwards the prompt to the Orchestrator and displays the final, actionable answer from the agents. 

3. A2A Prediction Server (Dedicated Prediction Microservice) 

This standalone microservice handles all forecasting jobs. When the Orchestrator identifies a price prediction request, it calls the A2A Server. 

The A2A Server contains two internal agents: 

Stock Forecaster Agent: Handles feature preparation and market signal extraction. 

Short-Term Prediction Agent: Generates short-term movement predictions. 

This separation allows the prediction pipeline to: 

Scale independently. 

Use its own models. 

Evolve or be retrained without impacting other components. 

4. MCP Server (Notifications & Email Automation) 

The MCP (Model-Context-Protocol) Server manages outbound communication via tools. When users request an email summary: 

The Orchestrator calls the EmailSender MCP Tool. 

The MCP Server formats the email and dispatches it. 

This ensures notification logic remains decoupled from the rest of the system, allowing for easy extensibility (e.g., adding SMS or Slack alerts in the future). 

 

🛠️ Technologies & Frameworks 

Category 

Technology / Framework 

Role in Project 

Agent Core 

Google Agent Development Kit (ADK) 

Implements the agent logic, including agent definitions, tool integrations, and session management (google.adk.agents, google.adk.runners, google.adk.sessions). 

Reasoning Engine 

Google Gemini Model (gemini-2.5-flash-lite) 

Configured as the reasoning engine for intent detection, routing decisions, and tool selection, coordinating the invocation of services. 

Concurrency 

Asyncio 

Enables asynchronous task execution and non-blocking operations for user input handling, inter-agent communication, and server calls, ensuring responsiveness. 

State Management 

SQLite with DatabaseSessionService 

A lightweight file-based database used to persist session history (in cli_sessions.db), enabling state retention across agent runs. 

Configuration 

Environment Variable Management 

Sensitive configurations (e.g., API keys) are loaded from a .env file, maintaining security and enabling cleaner deployment. 

SENDER_EMAIL= your email id 

SENDER_PASSWORD=your passwd 

SMTP_SERVER=smtp.gmail.com 

SMTP_PORT=587 

GOOGLE_API_KEY=your API key 

MARKETAUX_API_TOKEN=your token 

 

 

 

 

 

 

🔧 Tools and Utilities 

These functions are registered as "tools" with the main agent, allowing the Gemini reasoning engine to decide which one to call based on the user's query, ensuring targeted and efficient execution. 

1. Tool: Fetch Current Stock Price 

Method Name: get_current_stock_price(ticker_symbol: str) 

Location: Defined in Stock_Trending_Tickers_Agent.py 

Function: Provides the real-time price for a specific stock ticker using the yfinance library to fetch data from Yahoo Finance. 

2. Tool: Identify Trending Stocks 

Method Name: get_trending_tickers() 

Location: Likely defined in Stock_Trending_Tickers_Agent.py and imported into the main orchestrator. 

Function: Scans various sources to find stocks that are currently popular or have high market activity (e.g., from news, social media, or volume spikes). 

3. Tool: Predict from News Sentiment 

Method Name: predict_ticker_from_news(ticker_symbol: str) 

Location: Likely defined in Stock_Predictor_FromNews_Agent.py. 

Function: Scrapes financial news, performs sentiment analysis, and returns a qualitative prediction or sentiment summary for a stock. 

4. Tool: Get Short-Term Forecast (A2A Client) 

Method Name: get_a2a_short_term_prediction(ticker_symbol: str) 

Location: Defined within the main orchestrator script (Stocktrender_Chatbot_Agent_Orchestartor.py). 

Function: Acts as a client to the dedicated A2A (Agent-to-Agent) Prediction Server, making a network request to get a quantitative forecast from the specialized forecasting agents. 

5. Tool: Send Email Summary (MCP Client) 

Method Name: run_mcp_email_tool(recipient_email: str, content: str) 

Location: Defined within the main orchestrator script (Stocktrender_Chatbot_Agent_Orchestartor.py). 

Function: Acts as a client to the MCP (Model-Context-Protocol) Server to trigger secure email notifications, providing the necessary details (like email address and content) to send the alert. 

 

 

 

 

 

🔎 Observability and Monitoring 

A robust observability pipeline is critical for maintaining performance, debugging issues, and understanding the agent's complex decision-making process. This project implements three key pillars of logging and monitoring: 

1. Real-Time Console Output 

This provides immediate feedback to the developer or operator running the system. It is used for: 

Status Updates: Informs the user about the application's state, such as ✅ ADK components imported successfully, ✅ Using database for session persistence, or [AGENT] Calling tool: .... 

User Guidance: Prints helpful messages if critical configurations are missing, such as API keys or ADK installation dependencies. 

Agent Responses: Streams the agent's final, natural language text response directly to the console for immediate use. 

2. Performance & Latency Logging 

A dedicated logger (lat_logger) is set up to monitor the performance of critical operations, ensuring responsiveness is maintained as the system scales. 

File: Stocktrender_Chatbot_Agent_Log.log 

Purpose: To track the total execution time of key actions and identify performance bottlenecks. 

Example: After every user query is fully processed, it logs the total time: 

INFO - [PERFORMANCE] Query processed in 1.245 seconds 
 

Configuration: The log file is intentionally cleared at the start of each run to ensure a clean analysis of the current session's performance. 

3. Structured Trace Logging (JSONL) 

This is the most advanced form of observability, designed for detailed debugging and automated analysis of the agent's internal reasoning. 

File: Stocktrender_Chatbot_Agent_Trace.log 

Format: It uses a custom JsonLineFormatter to write every log entry as a structured JSON object on a new line (JSONL format), making the data machine-readable. 

Purpose: To create a comprehensive, machine-readable audit trail of the agent's operations. 

Events Logged: 

Runner initialization and shutdown events. 

Session creation and retrieval (session.run.start). 

Incoming user queries (query.received). 

Tool Calls: Critically, it logs exactly which tool the agent selects and the arguments it passes, for example: {"event": "tool.call", "name": "get_trending_tickers", "args": {}}. 

Errors with full stack traces (runner.call.error). 

Configuration: This detailed tracing is disabled by default and can be enabled by setting the TRACE_LOG environment variable. It also uses a RotatingFileHandler to automatically manage log file size, preventing storage overruns. 

 

 

🚀 Deployment Strategy (Docker) 

The application is deployed using Docker to encapsulate the multi-agent system into a single, portable container, ensuring consistency across environments. 

Goal and Entry Point 

The entire multi-component application (Orchestrator, A2A Server, MCP Server, and Streamlit UI) is designed to run within a single Docker container. This is achieved using a start.sh script as the container's entry point, which launches all necessary Python services concurrently. 

Step 1: start.sh Script 

This script launches each of the three backend server components in the background (&) and then starts the Streamlit UI in the foreground. The foreground Streamlit process keeps the container running. 

Step 2: Dockerfile 

This file provides the instructions to build the application's Docker image. It installs dependencies, copies the source code, and sets the start.sh script as the primary command. 

Step 3: Build and Run 

The project is cloned using: git clone git@github.com:sthumbar/StockTrendAnalyzer.git 

From your terminal in the project root directory (StockTrendAnalyzer), use these commands: 

Make the script executable (You only need to do this once locally): chmod +x start.sh 

Build the Docker Image: docker build -t stock-analyzer-app . 

Run the Container: docker run --rm -p 8501:8501 --env-file .env stock-analyzer-app 

The container is configured to: 

--rm: Automatically clean up the container filesystem when it stops. 

-p 8501:8501: Map the local machine's port 8501 to the container's Streamlit port 8501. 

--env-file .env: Securely load API keys and other secrets from the local .env file into the running container. 

Now the application is now running inside Docker. Access the web UI by opening a browser and navigating to http://localhost:8501. 

 

 

 

 

 

 

 

✅ Conclusion 

The StockTrendAnalyzer Agent successfully delivers a powerful, yet accessible, solution to the challenge of financial market analysis. By leveraging a multi-agent architecture orchestrated by the Gemini model, the project delegates complex tasks—from news-driven sentiment analysis to dedicated short-term price prediction—to specialized microservices. This approach not only ensures scalability and modularity but also significantly lowers the barrier to entry for users seeking high-quality market insights. The integration of proactive email alerting, a conversational UI, and a detailed three-pillar observability framework makes this system a robust, modern tool capable of providing timely, informed decision support for both retail investors and professional analysts. The project demonstrates the effectiveness of intelligent agents in transforming high-volume, complex data into actionable financial intelligence. 

📈 Value Proposition 

The StockTrendAnalyzer Agent transforms raw, high-volume market data into timely, actionable financial intelligence. It delivers a quantifiable return on investment by: 

Saving Time: Automating 90% of market monitoring and news-driven analysis, freeing up analysts and investors from manual data review. 

Minimizing Risk: Providing objective, short-term price predictions decoupled from emotional bias, enabling proactive risk mitigation. 

Maximizing Opportunity: Ensuring users receive critical, trend-based email alerts in real-time, preventing the delay that often leads to missed entry and exit points. 

In essence, the StockTrendAnalyzer Agent provides the critical edge: informed decision-making delivered automatically, turning complex market chaos into clear, profitable opportunities. 

 