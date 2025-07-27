#define tools
from langchain.agents import tool
from data_loader import load_cv, write_to_docx
from search import job_threads, get_job_ids
import asyncio
from langchain.pydantic_v1 import BaseModel, Field

@tool
def job_pipeline(keywords: str, location_name:str, job_type:str=None, limit:int=10, companies:str=None, industries:str=None, remote:str=None) -> dict:
    """Search LinkedIn for job postings based on specified criteria."""
    print(f"DEBUG: job_pipeline received location_name={location_name}, remote={remote}")
    job_ids = get_job_ids(keywords, location_name, job_type, limit, companies, industries, remote)
    print(f"DEBUG: job_pipeline job_ids={job_ids}")
    job_desc = asyncio.run(job_threads(job_ids))
    return job_desc

@tool
def extract_cv() -> dict:
    """Extract and structure job-relevant information from an uploaded CV."""
    cv_extracted_json = {}
    text = load_cv("tmp/cv.pdf")
    cv_extracted_json['content'] = text
    return cv_extracted_json

@tool
def generate_letter_for_specific_job(cv_details: str, job_details: str) -> str:
    """Generate a tailored cover letter using the provided CV and job details."""
    # This tool is called by the agent to generate content
    # The actual generation is handled by the LLM through the agent
    prompt = f"""
    Please generate a personalized cover letter and CV improvements based on:
    
    CV Content:
    {cv_details}
    
    Job Details:
    {job_details}
    
    Create a compelling cover letter that matches the job requirements with the candidate's experience
    and suggest specific improvements to make the CV more suitable for this position.
    """
    return prompt

def get_tools():
    return [job_pipeline, extract_cv, generate_letter_for_specific_job]

@tool
def func_alternative_tool(msg: str, members):
    """Router tool route message among different members"""
    members = ["Analyzer", "Generator", "Searcher"]
    options = ["FINISH"] + members
    # Using openai function calling can make output parsing easier for us
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    return function_def