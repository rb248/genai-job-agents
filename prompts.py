from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
llama3_begin_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> "
llama3_end_template = " <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"

def routing_prompt(llm_name, options, members):
      system_prompt = get_system_prompt(llm_name)

      if llm_name=='openai':
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("system",  "Given the conversation above, who should act next? Or is the task complete and should we FINISH?  Select one of: {options}"),
            ]).partial(options=str(options), members=", ".join(members))
      elif llm_name=='groq':
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                ("system",  llama3_begin_template + "Summarize and asses the conversation. Given the conversation above, who should act next? Or is the task complete and should we FINISH?  Select one of: {options}" + llama3_end_template),
            ]).partial(options=str(options), members=", ".join(members))
      else:
            # A simplified prompt for Gemini and other non-OpenAI models
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]).partial(options=str(options), members=", ".join(members))
      return prompt

def get_system_prompt(llm_name):
      if llm_name=='openai':
            SYSTEM_PROMPT = "You are a supervisor agent tasked with managing a conversation between the"\
            " following workers:  {members}. User has uploaded a document and sent a query. Given the uploaded document and following user request,"\
            " respond with the worker to act next. Each worker will perform a"\
            " task and respond with their results and status." \
            " only route the tasks based on the router if there is anything to route or task is not complete." \
            " When finished, respond with FINISH."
      elif llm_name=='groq':
            SYSTEM_PROMPT = llama3_begin_template + "You are a supervisor agent tasked with managing a conversation between the"\
            " following workers:  {members}. User has uploaded a CV and sent a query. Given the uploaded CV and following user request,"\
            " respond with the worker to act next. Each worker will perform a"\
            " task and respond with their results and status." \
            " After the result: ask yourself from the original query if the task is satisfied? based on that pass it to next appropriate route. " \
            " When task is finished, respond with FINISH." \
            + llama3_end_template
      else:
            # A highly restrictive prompt for Gemini and other non-OpenAI models
            SYSTEM_PROMPT = (
                "Your only job is to choose the next action from this list: {options}. "
                "Read the user's last message and the provided CV content, then decide the next step. "
                "Your response MUST be a single word from the list. Nothing else."
            )
      return SYSTEM_PROMPT


def get_search_agent_prompt(llm_name):
      json_instruction = """
Please output the job results as a JSON array of objects, where each object contains:
- job_title
- company_name
- job_location
- job_summary
- application_link
- company_url
Example:
[
  {{
    "job_title": "Data Scientist",
    "company_name": "Acme Corp",
    "job_location": "Berlin, Germany",
    "job_summary": "Work on ML models...",
    "application_link": "https://acme.com/apply",
    "company_url": "https://acme.com"
  }}
]
"""
      if llm_name=='openai':
            SEARCH_AGENT = (
                "Search for job listings based on user-specified parameters, including filters for remote jobs and specific locations. "
                "For each job, ALWAYS include job_title, company_name, job_location, job_summary, application_link, and company_url. "
                + json_instruction +
                "If unsuccessful, retry with alternative keywords up to three times and provide the results."
            )
      elif llm_name=='groq':
            SEARCH_AGENT = llama3_begin_template + (
                "You are a Searcher Agent. Search for job listings based on user-specified parameters, including filters for remote jobs and specific locations. "
                "For each job, ALWAYS include job_title, company_name, job_location, job_summary, application_link, and company_url. "
                + json_instruction +
                "If unsuccessful, retry with alternative keywords up to three times and provide the results."
            ) + llama3_end_template
      else:
            # Default/generic search agent prompt for all other models
            SEARCH_AGENT = (
                "Search for job listings based on user-specified parameters, including filters for remote jobs and specific locations. "
                "For each job, ALWAYS include job_title, company_name, job_location, job_summary, application_link, and company_url. "
                + json_instruction +
                "If unsuccessful, retry with alternative keywords up to three times and provide the results."
            )
      return SEARCH_AGENT

def get_analyzer_agent_prompt(llm_name):
      if llm_name=='openai':
            ANALYZER_AGENT = "Here is the user's CV:\n{cv_content}\n\nAnalyze the content of the user-uploaded CV above and matching job listings to recommend the best job fit, detailing the reasons behind the choice."
      elif llm_name=='groq':
            ANALYZER_AGENT = llama3_begin_template + "You are an Analyzer Agent. \
            Here is the user's CV:\n{cv_content}\n\nAnalyze the content of the user-uploaded CV above and matching job listings to recommend the best job fit, \
            detailing the reasons behind the choice." \
            + llama3_end_template
      else:
            # Default/generic analyzer agent prompt for all other models
            ANALYZER_AGENT = "Here is the user's CV:\n{cv_content}\n\nAnalyze the content of the user-uploaded CV above and matching job listings to recommend the best job fit, detailing the reasons behind the choice."
      return ANALYZER_AGENT

def get_generator_agent_prompt(llm_name):
      base_prompt = """You are a Generator Agent specialized in creating personalized cover letters and improving CVs based on specific job requirements.

Your tasks:
1. Generate a tailored cover letter that matches the job requirements with the candidate's experience
2. Suggest improvements to the CV to better align with the job requirements

Use the CV content and job details provided in the conversation to create:
- A compelling cover letter that highlights relevant experience and skills
- Specific suggestions for CV improvements

Always structure your response clearly with sections for:
- COVER LETTER: [generated cover letter]
- CV IMPROVEMENTS: [suggested improvements]

Make sure to:
- Address the specific requirements mentioned in the job posting
- Highlight relevant experience from the CV
- Use professional language and formatting
- Keep the cover letter concise but impactful (3-4 paragraphs)"""

      if llm_name=='openai':
            GENERATOR_AGENT = base_prompt
      elif llm_name=='groq':
            GENERATOR_AGENT = llama3_begin_template + base_prompt + llama3_end_template
      else:
            # Default/generic generator agent prompt for all other models
            GENERATOR_AGENT = base_prompt
      return GENERATOR_AGENT

### Example input:
#Find data science job for me in Germany maximum 5 relevant one. \
# Then analyze my CV and write me a cover letter according to the best matching job.