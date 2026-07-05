from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.messages import HumanMessage

#loading environment variables from .env file
load_dotenv(override=True)  # Load environment variables from .env file

#defining the model
model = ChatOpenRouter(model_name='nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free', temperature=0.1, max_tokens=1064)

#defining the search tool
@tool("search_tool", description="A tool for searching the web for information.")
def search_tool(query: str) -> Dict[str, Any]:
    return TavilyClient().search(query)


#defining system prompt
system_prompt = """
You are an indian chef who is an expert in cooking.
You will be use search tool to find the best recipe for the ingredients user has and provide them with a step by step guide to cook the dish.
"""

#creating the agent
agent = create_agent(model=model, tools=[search_tool], system_prompt=system_prompt)

query = HumanMessage(content="I have soyachunks, ginger garlic paste, tomatoes and onions, basic indian spices and cooked rice in fridge. give me one or two recipes for easy to make healthy food items using these ingredients.")


response = agent.invoke({"messages": query})

print(response["messages"])




