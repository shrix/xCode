# export OPENAI_API_KEY = '...'
# export GEMINI_API_KEY = '...'

import os
import streamlit as st
# from gemini.client import GeminiClient
from langchain_community.chat_models import Llama3Chat, GroqChat
from openai.api_resources import Completion
from openai.api_keys import api_key

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Set up OpenAI API key
api_key(OPENAI_API_KEY)

# Initialize language models
# gemini_client = GeminiClient(GEMINI_API_KEY)
# gemini_llm = lambda query: gemini_client.complete(query)
llama3_groq_llm = Llama3Chat(model_name="llama3-base", temperature=0.7, device="groq")
openai_llm = Completion()
openai_llm.model = "gpt-3.5-turbo"
openai_llm.temperature = 0.7

# Define the agents
OpenAI_Agent = Agent(
    role="OpenAI Agent",
    goal="Provide a response to the user's query using the OpenAI language model",
    backstory="I am an AI agent that uses the OpenAI language model to generate responses to user queries.",
    llm=openai_llm
)

# Gemini_Agent = Agent(
#     role="Gemini Agent",
#     goal="Provide a response to the user's query using the Gemini language model",
#     backstory="I am an AI agent that uses the Gemini language model to generate responses to user queries.",
#     llm=gemini_llm
# )

Llama_Agent = Agent(
    role="Llama3 Agent",
    goal="Provide a response to the user's query using the Llama3 language model",
    backstory="I am an AI agent that uses the Llama3 language model to generate responses to user queries.",
    llm=llama3_groq_llm
)

# Define the task
task = Task(
    description="Provide a response to the user's query",
    agent=OpenAI_Agent
)

# Define the crew
crew = Crew(
    agents=[OpenAI_Agent, Llama_Agent],
    tasks=[task]
)

# Define the function to blend the responses
def blend_responses(responses):
    return "\n\n".join(responses)

# Define the Streamlit app
def main():
    st.title("Chatbot")
    query = st.text_input("Enter your query:")
    if query:
        # Pass the query to the crew and collect the responses
        responses = crew.kickoff(query)
        # Blend the responses
        blended_response = blend_responses(responses)
        # Display the blended response
        st.write(blended_response)

if __name__ == "__main__":
    main()