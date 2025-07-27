#define LLMs - Enhanced with modern models similar to VS Code LM API capabilities
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

def load_llm(llm_name): 
    """
    Load language models with capabilities similar to VS Code LM API.
    Supports: GPT-4o, GPT-4o-mini, Claude-3.5-sonnet, Gemini 2.5, Groq models, and local models.
    """
    
    if llm_name == 'gpt-4o':
        # Latest GPT-4o model - similar to what VS Code LM API provides
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=os.environ["OPENAI_API_KEY"], 
            temperature=0.1, 
            streaming=True
        )
    
    elif llm_name == 'gpt-4o-mini':
        # GPT-4o-mini for faster, cost-effective operations
        llm = ChatOpenAI(
            model="gpt-4o-mini", 
            openai_api_key=os.environ["OPENAI_API_KEY"], 
            temperature=0.1, 
            streaming=True
        )
    
    elif llm_name == 'openai':
        # Legacy GPT-4 model
        llm = ChatOpenAI(
            model="gpt-4-0125-preview", 
            openai_api_key=os.environ["OPENAI_API_KEY"], 
            temperature=0.1, 
            streaming=True
        )
    
    elif llm_name == 'claude-3.5-sonnet':
        # Claude 3.5 Sonnet - excellent for reasoning and code tasks
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            temperature=0.1,
            streaming=True
        )
    
    elif llm_name == 'gemini-2.5-pro':
        # Gemini 2.5 Pro - Google's most powerful thinking model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            temperature=0.1
        )
    
    elif llm_name == 'gemini-2.5-flash':
        # Gemini 2.5 Flash - Best price-performance with adaptive thinking
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            temperature=0.1
        )
    
    elif llm_name == 'gemini-2.5-flash-lite':
        # Gemini 2.5 Flash-Lite - Most cost-efficient for high throughput
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            temperature=0.1
        )
    
    elif llm_name == 'groq':
        # Groq models for fast inference
        llm = ChatGroq(
            temperature=0.2, 
            groq_api_key=os.environ["GROQ_API_KEY"], 
            model_name="llama3-70b-8192"
        )
    
    elif llm_name == 'groq-mixtral':
        # Groq Mixtral model
        llm = ChatGroq(
            temperature=0.2, 
            groq_api_key=os.environ["GROQ_API_KEY"], 
            model_name="mixtral-8x7b-32768"
        )
    
    elif llm_name == "llama3-local":
        # Local Ollama model
        llm = ChatOpenAI(
            model="llama3", 
            base_url="http://localhost:11434/v1", 
            temperature=0.0,
            api_key="not-needed"  # Ollama doesn't require API key
        )
    
    else:
        # Default to GPT-4o if no valid model specified
        print(f"Warning: Unknown model '{llm_name}', defaulting to gpt-4o")
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=os.environ["OPENAI_API_KEY"], 
            temperature=0.1, 
            streaming=True
        )
    
    return llm