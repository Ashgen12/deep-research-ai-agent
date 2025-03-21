# Deep Research AI Agent for the Indian Stock Market 🤖

This project is an AI-powered research ai agent which is designed to help users analyze and gather insights about the Indian stock market.  It leverages the Deepseek API, LangChain, and various tools to fetch real-time data, summarize information, and answer user queries related to the Indian stock market.

## Features

*   **Fetch 52-week high stocks:**  Quickly identify top-performing stocks in the Indian market that are currently near their 52-week high.
*   **Gather Information:** Collect relevant data from diverse sources, including Google News, about specific topics or companies.
*   **Summarize Insights:** Utilize the Deepseek LLM to concisely summarize key insights extracted from the gathered information.
*   **Interactive UI:** A clean and user-friendly Gradio interface provides seamless interaction with the agent.
*   **Analyze Stock Performance:** Provides capability to get detailed information on specified stocks.

## Tech Stack

*   **Deepseek API:** For LLM, reasoning, and summarization capabilities.
*   **LangChain:** Framework for building and managing the agent, orchestrating tool interactions.
*   **Gradio:**  For creating the interactive web application interface.
*   **NSE API :**  For fetching real-time stock data from the National Stock Exchange of India. 
*   **Web Scraping (Beautiful Soup):** For gathering information from external web sources like Google News.
*   **Requests:** For making HTTP requests.

## How It Works

1.  **User Input:** The user poses a question or request through the Gradio interface (e.g., "What are the 52-week high stocks in the Indian market?").
2.  **Agent Execution:** The LangChain agent receives the user's input and determines the appropriate tools to use.  This might involve a combination of `get_52_week_high_stocks`, `gather_information`, and `summarize_insights`.
3.  **Tool Execution:** The selected tools perform their specific tasks:
    *   `get_52_week_high_stocks`: Fetches 52-week high stock data from the NSE.
    *   `gather_information`: Scrapes relevant information from sources like Google News based on the query.
    *   `summarize_insights`: Uses the Deepseek LLM to process and summarize the collected data into a coherent response.
4.  **Response:** The agent presents a structured and detailed response to the user through the Gradio interface, based on the gathered and summarized data.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Ashgen12/deep-research-ai-agent.git
    cd deep-research-ai-agent
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Obtain a Deepseek API Key:**  You'll need a valid API key from Deepseek to use their LLM services. Sign up for an account on their website and obtain your key.

4.  **Set your Deepseek API Key:**  Replace `"your-api-key-here"` in your code (likely in `main.py` or a configuration file) with your actual Deepseek API key:

    ```python
    # Example (main.py or config.py):
    DEEPSEEK_API_KEY = "your-actual-deepseek-api-key"
    ```
     It is a good practice to use environment variables instead of hardcoding the API Key.  If so, instructions should reflect that:

    ```bash
    # Set the environment variable (example for bash)
    export DEEPSEEK_API_KEY="your-actual-deepseek-api-key"
    ```
    And the code should read:
    ```python
    import os
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
    ```

5.  **Run the application:**

    ```bash
    python main.py
    ```

6.  **Access the UI:**  Open the Gradio interface in your web browser (usually at `http://127.0.0.1:7860` or a similar local address provided in the console output).


## Requirements

*   Python 3.8+
*   Libraries (listed in `requirements.txt`)

## Acknowledgments

*   **Deepseek:** For providing the powerful LLM API.
*   **LangChain:** For simplifying agent creation and tool management.
*   **Gradio:** For enabling a seamless and interactive user interface.
*   **NSE (National Stock Exchange of India):** For providing stock market data.
*   **Developers of `nsepy` (or the chosen NSE library):** For creating tools to interact with the NSE API.

## Contact

For questions or feedback, feel free to reach out:

*   📧 Email: ashunaukari01@gmail.com
*   🌐 GitHub: Ashgen12

Enjoy using the Deep Research AI Agent for the Indian Stock Market! 🚀
