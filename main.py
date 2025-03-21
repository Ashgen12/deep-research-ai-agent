# Description: A Deep Research Agent for the Indian stock market that can answer questions, gather information, and summarize key insights using the Deepseek API and various sources.

import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict
from langchain.agents import AgentOutputParser
from langchain.chains import LLMChain
import re
import gradio as gr

# Custom prompt template for the agent
class StockAnalysisPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.log}\nObservation: {observation}\n"
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["thoughts"] = thoughts
        return self.template.format(**kwargs)

# Initialize Deepseek API configuration
def load_model():
    DEEPSEEK_API_KEY = " "  # Replace with your Deepseek Openrouter API key
    DEEPSEEK_API_BASE = "https://openrouter.ai/api/v1"
    
    llm = ChatOpenAI(
        model_name="deepseek/deepseek-r1-distill-qwen-1.5b",
        temperature=0,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_API_BASE,
        streaming=False
    )
    return llm

# Fetch 52-week high stocks from NSE API
def get_52_week_high_stocks():
    """Fetch 52-week high stocks in the Indian stock market using the NSE API."""
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract stocks that are near their 52-week high
        stocks = []
        for item in data.get("data", []):
            if item.get("high52") and item.get("lastPrice"):
                if abs(float(item["high52"]) - float(item["lastPrice"])) / float(item["high52"]) < 0.05:  # Within 5% of 52-week high
                    stocks.append(item["symbol"])
        
        return stocks[:10]  # Return top 10 stocks near their 52-week high
    except Exception as e:
        return f"Error fetching data from NSE API: {e}"

# Gather relevant information from various sources
def gather_information(topic: str):
    """Gather relevant information from various sources based on the research topic."""
    base_url = f"https://www.google.com/search?q={topic}+Indian+stock+market"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="SoaBEf")
    info = [article.text for article in articles]
    return info

# Summarize key insights using Deepseek LLM
def summarize_insights(info: List[str]):
    """Summarize key insights from the gathered information."""
    llm = load_model()
    summary = llm.predict("\n".join(info))
    return summary

# Create the Deep Research Agent
def create_research_agent(llm):
    tools = [
        Tool(
            name="get_52_week_high_stocks",
            func=get_52_week_high_stocks,
            description="Fetch 52-week high stocks in the Indian stock market for the last month."
        ),
        Tool(
            name="gather_information",
            func=gather_information,
            description="Gather relevant information from various sources based on the research topic."
        ),
        Tool(
            name="summarize_insights",
            func=summarize_insights,
            description="Summarize key insights from the gathered information."
        )
    ]
    
    template = """You are a Deep Research Agent. Your goal is to help users perform deep research on the Indian stock market.

                You have access to the following tools:
                {tools}

                Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Previous conversation:
                {thoughts}

                Question: {input}"""

    prompt = StockAnalysisPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    
    class ResearchAgentOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            
            if not match:
                return AgentFinish(
                    return_values={"output": "I cannot determine what action to take. Please try rephrasing your question."},
                    log=llm_output,
                )
                
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
    output_parser = ResearchAgentOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    
    memory = ConversationBufferMemory(memory_key="thoughts")
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

# Initialize the agent
llm = load_model()
agent = create_research_agent(llm)

# Gradio UI
def ask_question(question):
    try:
        response = agent.run({"input": question})
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Beautify the UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Deep Research Agent for Indian Stock Market")
    gr.Markdown("Ask me anything about the Indian stock market!")
    
    with gr.Row():
        question_input = gr.Textbox(label="Your Question", placeholder="e.g., What are the 52-week high stocks in the Indian market?")
        submit_button = gr.Button("Ask")
    
    output = gr.Textbox(label="Response", interactive=False)
    
    gr.Markdown("### Example Questions")
    gr.Markdown("""
    - What are the 52-week high stocks in the Indian market?
    - Gather information about the recent trends in the Indian stock market.
    - Summarize the key insights about the Indian stock market.
    - Analyze the performance of the top 5 stocks in the Indian market.
    """)
    
    submit_button.click(ask_question, inputs=question_input, outputs=output)

# Launch the Gradio app
demo.launch()