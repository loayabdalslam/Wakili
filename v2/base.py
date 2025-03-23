from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
import google.genai as genai
from google.genai import types
import json
from colorama import init, Fore, Style
import logging
from datetime import datetime
import time

# Initialize colorama for colored terminal output
init()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentCrew")

# Load environment variables
load_dotenv()

# Define a simple Gemini LLM wrapper with delay
class GeminiLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.last_call_time = 0

    def __call__(self, messages):
        # Enforce 30-second delay between calls
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < 10:
            sleep_time = 10 - time_since_last
            logger.info(f"{Fore.YELLOW}Waiting {sleep_time:.1f} seconds before next API call...{Style.RESET_ALL}")
            time.sleep(sleep_time)
        
        try:
            contents = []
            for msg in messages:
                if msg.type == "human":
                    role = "user"
                    content = msg.content
                elif msg.type == "system":
                    role = "user"
                    content = f"System instruction: {msg.content}"
                else:
                    continue
                contents.append({"role": role, "parts": [{"text": content}]})
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.95,
                    top_k=64,
                    max_output_tokens=8192,
                    response_mime_type="text/plain",
                )
            )
            self.last_call_time = time.time()  # Update last call time
            return response.text
        except Exception as e:
            self.last_call_time = time.time()  # Update even on error
            return f"Error calling Gemini API: {str(e)}"

# Define the response model (for parsing intermediate JSON if needed)
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    people: list[str]
    organizations: list[str]
    events: list[str]
    places: list[str]
    speed_of_response: str
    accuracy_of_response: str
    completeness_of_response: str
    relevance_of_response: str
    clarity_of_response: str
    creativity_of_response: str
    depth_of_response: str

# Define tools
@tool
def web_search(query: str) -> str:
    """Simulates a web search for the given query."""
    return f"Web search results for '{query}': [Simulated data]"

@tool
def save_to_file(content: str) -> str:
    """Saves the given content to a Markdown file."""
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Content saved to {filename}"

tools = {"web_search": web_search, "save_to_file": save_to_file}

# Define Agent class
class Agent:
    def __init__(self, name, role, llm, prompt_template):
        self.name = name
        self.role = role
        self.llm = llm
        self.prompt = prompt_template

    def process(self, message, crew_log):
        logger.info(f"{Fore.CYAN}{self.name} ({self.role}) processing: {message}{Style.RESET_ALL}")
        messages = self.prompt.format_messages(message=message)
        response = self.llm(messages)
        crew_log.append(f"### {self.name} ({self.role})\n\n{response}")
        return response

# Set up the LLM and parser
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
llm = GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-pro"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define agent prompts
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Researcher agent. Gather information on the query.
        Available tools: {tool_names}.
        If a tool is needed, include '[tool_name: query]' in your response.
        Provide detailed findings for the Writer agent.
    """),
    ("human", "{message}")
]).partial(tool_names="web_search")

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a Writer agent. Take the Researcher's input and create a detailed report in Markdown format.
        Available tools: {tool_names}.
        Structure the report with:
        # Report: <topic>
        ## Summary
        <summary text>
        ## Key Details
        - **Sources**: <list sources>
        - **Tools Used**: <list tools>
        - **People**: <list people>
        - **Organizations**: <list organizations>
        - **Events**: <list events>
        - **Places**: <list places>
        ## Evaluation
        - **Speed**: <speed>
        - **Accuracy**: <accuracy>
        - **Completeness**: <completeness>
        - **Relevance**: <relevance>
        - **Clarity**: <clarity>
        - **Creativity**: <creativity>
        - **Depth**: <depth>
        After preparing the report, include '[save_to_file: <full_markdown_report>]' to save it.
    """),
    ("human", "{message}")
]).partial(tool_names="save_to_file")

# Create agents
researcher = Agent("Researcher", "Information Gatherer", llm, researcher_prompt)
writer = Agent("Writer", "Response Formatter", llm, writer_prompt)

# Custom crew executor
def execute_crew(query):
    crew_log = [f"# Research Crew Report\n\n**Query:** {query}\n\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    logger.info(f"{Fore.GREEN}Crew starting work on: {query}{Style.RESET_ALL}")

    # Researcher processes the query
    research_result = researcher.process(query, crew_log)
    while "[web_search:" in research_result:
        start_idx = research_result.index("[web_search:") + 12
        end_idx = research_result.index("]", start_idx)
        tool_query = research_result[start_idx:end_idx]
        tool_result = tools["web_search"](tool_query)
        logger.info(f"{Fore.YELLOW}Researcher using web_search: {tool_query} -> {tool_result}{Style.RESET_ALL}")
        research_result = researcher.process(f"Tool result: {tool_result}", crew_log)

    # Writer processes the research result
    final_response = writer.process(research_result, crew_log)
    if "[save_to_file:" in final_response:
        start_idx = final_response.index("[save_to_file:") + 14
        end_idx = final_response.index("]", start_idx)
        content_to_save = final_response[start_idx:end_idx]
        tool_result = tools["save_to_file"](content_to_save)
        logger.info(f"{Fore.MAGENTA}Writer saving to file: {tool_result}{Style.RESET_ALL}")

    # Fallback save and return Markdown
    markdown_output = "\n\n".join(crew_log)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    logger.info(f"{Fore.MAGENTA}Fallback: Saved output to {filename}{Style.RESET_ALL}")
    logger.info(f"{Fore.GREEN}Crew completed{Style.RESET_ALL}")
    return markdown_output

# Get user input and run the crew
query = input("What can I help you research? ")
output = execute_crew(query)
print(output)