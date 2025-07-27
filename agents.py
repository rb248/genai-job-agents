from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import operator
from typing import Annotated, Sequence, TypedDict, List
import functools
from langgraph.graph import StateGraph, END
from tools import *
from prompts import *
import os

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools) # type: ignore
    return executor

def agent_node(state, agent, name):
    print("STATES ARE", state)
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]} 

def define_graph(llm, llm_name):
    members = ["Analyzer", "Generator", "Searcher"]
    options = ["FINISH"] + members
    
    # Get the raw prompt string and format it with dynamic values
    raw_system_prompt = get_system_prompt(llm_name)
    system_prompt = raw_system_prompt.format(options=str(options), members=", ".join(members))
    
    llm_name=os.environ.get('LLM_NAME')
    # The prompt is now created with the fully formatted system_prompt
    prompt = routing_prompt(llm_name, options, members)

    print(llm_name)
    
    # Use the simple fallback approach for all models - it's more reliable
    def supervisor_chain_fallback(state):
        # Extract the list of BaseMessages from the state dict
        msgs = state["messages"] if "messages" in state else []
        # Ensure the system prompt is included as the first message
        system_prompt_msg = SystemMessage(content=system_prompt)
        msgs = [system_prompt_msg] + msgs

        response = llm.invoke(msgs)

        raw_content = response.content
        print(f"DEBUG: Raw supervisor response: '{raw_content}'")

        clean_content = "".join(c for c in raw_content if c.isalnum() or c.isspace()).lower()

        synonym_map = {
            "askforcv": "Analyzer",
            "ask_for_cv": "Analyzer",
            "cv": "Analyzer",
            "summary": "Analyzer",
            "summarize": "Analyzer",
            "summarise": "Analyzer",
            "analyze": "Analyzer",
            "analyse": "Analyzer",
            "analyzer": "Analyzer",
            "generator": "Generator",
            "searcher": "Searcher",
            "finish": "FINISH"
        }

        next_agent = None
        for key, value in synonym_map.items():
            if key in clean_content:
                next_agent = value
                break

        if not next_agent:
            print(f"DEBUG: Supervisor could not find a valid next agent in response. Defaulting to FINISH.")
            next_agent = "FINISH"

        print(f"DEBUG: Supervisor decided next agent is: {next_agent}")
        # Preserve all fields in state, just update/add 'next'
        new_state = dict(state)
        new_state["next"] = next_agent
        print("DEBUG: Supervisor returning state:", new_state)
        return new_state
    
    supervisor_chain = supervisor_chain_fallback


    import asyncio
    def search_node(state):
        #print("DEBUG: search_node received state:", state)
        # --- LLM-based search agent logic (commented out) ---
        # --- LLM-based search agent logic (commented out) ---
        # # Run the agent and get job results
        # # Map selected_cities and remote_only to job_pipeline args
        # # Set location_name and remote in state for the agent
        # state["location_name"] = state.get("selected_cities") or state.get("input") or "Germany"
        # state["remote"] = state.get("remote_only", False)
        #
        # # --- AGENT CONTROL FIX ---
        # # Force the agent to see the filters by modifying the input query itself.
        # original_query = state["input"]
        # locations = state.get("selected_cities")
        # is_remote = state.get("remote_only")
        #
        # # Build a new, more explicit query for the agent
        # new_query = f"Find jobs based on the query: '{original_query}'."
        # if locations:
        #     new_query += f" CRITICAL: You MUST search in these locations ONLY: {locations}."
        # if is_remote:
        #     new_query += f" CRITICAL: You MUST search for remote jobs ONLY."
        #
        # # Overwrite the input in the state that the agent will see
        # state["input"] = new_query
        # state["messages"] = [HumanMessage(content=new_query)]
        # # --- END AGENT CONTROL FIX ---
        #
        # print(f"DEBUG: search_node state={state}")
        # agent = create_agent(llm, [job_pipeline], get_search_agent_prompt(llm_name))
        # result = agent.invoke(state)
        # print("DEBUG: Raw LLM output from agent.invoke =", result)
        # jobs = result.get("output", [])
        # # If jobs is a string, try to parse as list
        # if isinstance(jobs, str):
        #     import json, re
        #     # Remove Markdown code fences if present
        #     jobs_clean = re.sub(r"^```(?:json)?|```$", "", jobs.strip(), flags=re.MULTILINE).strip()
        #     # Remove invalid escape sequences (e.g., stray backslashes not followed by valid escape char)
        #     # Only allow valid escapes: \", \\, \/, \b, \f, \n, \r, \t, \u
        #     jobs_clean = re.sub(r'\\(?!["\\/bfnrtu])', '', jobs_clean)
        #     try:
        #         jobs = json.loads(jobs_clean)
        #     except Exception as e:
        #         print("ERROR: Could not parse jobs JSON after cleaning. String was:\n", jobs_clean)
        #         print("Exception:", e)
        #         jobs = []
        #
        # # Store raw job results in the state
        # state["job_results"] = jobs
        #
        # # Format job results with links
        # formatted = []
        # for job in jobs:
        #     if isinstance(job, dict):
        #         title = job.get("job_title", "Unknown Title")
        #         company = job.get("company_name", "Unknown Company")
        #         location = job.get("job_location", "Unknown Location")
        #         job_url = job.get("application_link") or job.get("company_url") or ""
        #         desc = job.get("job_desc_text", "")
        #         line = f"**{title}** at {company} ({location})\n"
        #         if job_url:
        #             line += f"[Apply/Details]({job_url})\n"
        #         if desc:
        #             line += f"{desc[:200]}...\n"
        #         formatted.append(line)
        #     else:
        #         formatted.append(str(job))
        # output = "\n---\n".join(formatted) if formatted else "No jobs found."
        # return {
        #     "messages": [HumanMessage(content=output, name="Searcher")],
        #     "job_results": jobs
        # }
        # --- Direct API job search logic ---
        keywords = state.get("input", "")
        locations = state.get("selected_cities") or ["Germany"]
        remote = state.get("remote_only", False)
        job_type = state.get("job_type", None)
        limit = state.get("job_limit", 10)
        companies = state.get("companies", None)
        industries = state.get("industries", None)

        print(f"DEBUG: Direct API job search: keywords={keywords}, locations={locations}, remote={remote}, job_type={job_type}, limit={limit}, companies={companies}, industries={industries}")

        # Get job IDs for all locations
        from search import get_job_ids, job_threads
        job_ids = get_job_ids(keywords, locations, job_type, limit, companies, industries, remote)
        jobs = asyncio.run(job_threads(job_ids))

        # Store raw job results in the state
        state["job_results"] = jobs

        # Format job results with links
        formatted = []
        for job in jobs:
            if isinstance(job, dict):
                title = job.get("job_title", "Unknown Title")
                company = job.get("company_name", "Unknown Company")
                location = job.get("job_location", "Unknown Location")
                job_url = job.get("company_apply_url") or job.get("company_url") or ""
                desc = job.get("job_desc_text", "")
                line = f"**{title}** at {company} ({location})\n"
                if job_url:
                    line += f"[Apply/Details]({job_url})\n"
                if desc:
                    line += f"{desc[:200]}...\n"
                formatted.append(line)
            else:
                formatted.append(str(job))
        output = "\n---\n".join(formatted) if formatted else "No jobs found."
        return {
            "messages": [HumanMessage(content=output, name="Searcher")],
            "job_results": jobs
        }

    # Analyzer agent prompt now expects cv_content as a variable
    analyzer_prompt = get_analyzer_agent_prompt(llm_name).format(cv_content="{cv_content}")
    # Analyzer does not need tools unless you want function calling
    analyzer_agent = create_agent(llm, [], analyzer_prompt)
    def analyzer_node(state):
        # Fill in the cv_content in the prompt at runtime
        filled_prompt = get_analyzer_agent_prompt(llm_name).format(cv_content=state.get("cv_content", ""))
        agent = create_agent(llm, [], filled_prompt)
        print("DEBUG: Analyzer agent prompt:", filled_prompt)
        result = agent.invoke(state)
        return {"messages": [HumanMessage(content=result["output"], name="Analyzer")]}

    generator_agent = create_agent(llm, [generate_letter_for_specific_job], get_generator_agent_prompt(llm_name))
    generator_node = functools.partial(agent_node, agent=generator_agent, name="Generator")

    workflow = StateGraph(AgentState)
    workflow.add_node("Analyzer", analyzer_node)
    workflow.add_node("Searcher", search_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("supervisor", supervisor_chain)

    for member in members:
        # We want our workers to ALWAYS "report back" to the supervisor when done
        workflow.add_edge(member, "supervisor")
    # The supervisor populates the "next" field in the graph state
    # which routes to a node or finishes
    conditional_map = {k: k for k in members}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # Finally, add entrypoint
    workflow.set_entry_point("supervisor")

    graph = workflow.compile()
    return graph

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always be added to the current states
    input: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    cv_content: str
    selected_cities: List[str]
    remote_only: bool
    job_results: List[dict]
    job_limit: int

def generate_cv_and_cover_letter(llm, llm_name, cv_content, job_details):
    """
    Utility to invoke the Generator agent directly with CV and job details.
    Returns the generated cover letter and improved CV (if available).
    """
    generator_prompt = get_generator_agent_prompt(llm_name)
    generator_agent = create_agent(llm, [generate_letter_for_specific_job], generator_prompt)
    
    # Format job details for better context
    job_summary = f"""
Job Title: {job_details.get('job_title', 'N/A')}
Company: {job_details.get('company_name', 'N/A')}
Location: {job_details.get('job_location', 'N/A')}
Job Description: {job_details.get('job_desc_text', 'No description available')}
"""
    
    # Create a comprehensive message that includes both CV and job details
    message_content = f"""
Please generate a personalized cover letter and CV improvements for this job opportunity:

{job_summary}

Using this CV content:
{cv_content}

Please provide:
1. A tailored cover letter that highlights relevant experience
2. Specific suggestions for improving the CV for this role
"""
    
    state = {
        "cv_content": cv_content,
        "job_details": job_details,
        "input": message_content,
        "messages": [HumanMessage(content=message_content)]
    }
    
    print(f"DEBUG: Invoking generator agent with CV length: {len(cv_content)} chars, job: {job_details.get('job_title', 'Unknown')}")
    result = generator_agent.invoke(state)
    
    output = result.get("output", "No output generated")
    print(f"DEBUG: Generator agent output: {output[:200]}...")
    
    # Try to extract cover letter and CV improvements if structured
    if "COVER LETTER:" in output and "CV IMPROVEMENTS:" in output:
        parts = output.split("CV IMPROVEMENTS:")
        cover_letter = parts[0].replace("COVER LETTER:", "").strip()
        improved_cv = parts[1].strip()
    else:
        cover_letter = output
        improved_cv = "No specific CV improvements provided."
    
    return cover_letter, improved_cv
