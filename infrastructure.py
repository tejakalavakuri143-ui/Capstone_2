import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from langfuse.langchain import CallbackHandler  

load_dotenv() 

langfuse_handler = CallbackHandler()
def get_central_llm(model_name: str = "bedrock/cohere.command-r-plus-v1:0", temperature: float = 0.0):
    """
    Module 4 Gateway: All teams MUST call this function to get their LLM.
    """
    # ChatLiteLLM will automatically find OPENAI_API_KEY (or GEMINI_API_KEY) in the environment
    llm = ChatLiteLLM(
        model=model_name,
        temperature=temperature,
        callbacks=[langfuse_handler] 
    )
    return llm