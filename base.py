from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
import google.genai as genai
from google.genai import types

# Load environment variables
load_dotenv()

# Define a simple Gemini LLM wrapper
class GeminiLLM:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def __call__(self, messages):
        try:
            # Convert LangChain messages to Gemini format
            contents = []
            for msg in messages:
                # Map roles: "human" -> "user", "system" -> prepend to first user message
                if msg.type == "human":
                    role = "user"
                    content = msg.content
                elif msg.type == "system":
                    role = "user"  # Treat system as user since Gemini doesn't support "system"
                    content = f"System instruction: {msg.content}"
                else:
                    continue  # Skip unsupported message types
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
            return response.text
        except Exception as e:
            return f"Error calling Gemini API: {str(e)}"

# Define the response model
class ResearchResponse(BaseModel):
    # Define the fields for the research response
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    # Define the fields for the research response
    people: list[str]
    organizations: list[str]
    events: list[str]
    places: list[str]
    # Evaluation metrics
    speed_of_response: str
    accuracy_of_response: str
    completeness_of_response: str  
    relevance_of_response: str
    clarity_of_response: str
    creativity_of_response: str
    depth_of_response: str
# Define a sample tool (e.g., web search simulation)


@tool
def web_search(message: str) -> str:
    """Simulates a web search for the given query."""
    return f"Web search results for '{message}': [Simulated data]"

tools = {"web_search": web_search}

# Set up the LLM and parser
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
llm = GeminiLLM(model_name=os.getenv("GEMINI_MODEL", "gemini-pro"))
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant.
            Answer the user query and use necessary tools. Available tools: {tool_names}.
            If a tool is needed, include '[tool_name: query]' in your response.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("human", "{message}"),
    ]
).partial(
    format_instructions=parser.get_format_instructions(),
    tool_names=", ".join(tools.keys())
)


# Custom executor to handle tool calls
def execute_agent(message):
    # Initial message
    messages = prompt.format_messages(message=message)
    response = llm(messages)
    
    # Check for tool calls in the response
    while "[web_search:" in response:
        start_idx = response.index("[web_search:") + 12
        end_idx = response.index("]", start_idx)
        tool_query = response[start_idx:end_idx]
        tool_result = tools["web_search"](tool_query)
        
        # Append tool result to messages and re-run
        messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
        response = llm(messages)
    
    # Parse the final response
    try:
        parsed_response = parser.parse(response)
        return parsed_response.json()
    except Exception as e:
        return f"Error parsing response: {str(e)}"

# Get user input and run the query
query = input("What can I help you research? ")  # Fixed typo "reseaarch" to "research"
output = execute_agent(query)
print(output)