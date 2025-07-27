import streamlit as st
from dotenv import load_dotenv
import os
import time
from langchain_core.messages import AIMessage
load_dotenv()
from agents import define_graph, generate_cv_and_cover_letter
from llms import load_llm 
from langchain_core.messages import HumanMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from streamlit_pills import pills
from data_loader import load_cv
from utils import convert_to_csv, get_job_details_for_export

st.set_page_config(layout="wide")
st.title("GenAI Job Agent - 🦜")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Model Selection in Sidebar
st.sidebar.header("🤖 AI Model Configuration")
model_options = {
    'GPT-4o (Latest)': 'gpt-4o',
    'GPT-4o Mini (Fast & Cost-effective)': 'gpt-4o-mini', 
    'GPT-4 Turbo (Legacy)': 'openai',
    'Claude 3.5 Sonnet (Reasoning)': 'claude-3.5-sonnet',
    'Gemini 2.5 Pro (Google\'s Best)': 'gemini-2.5-pro',
    'Gemini 2.5 Flash (Adaptive Thinking)': 'gemini-2.5-flash',
    'Gemini 2.5 Flash-Lite (Cost Efficient)': 'gemini-2.5-flash-lite',
    'Groq Llama3 (Ultra Fast)': 'groq',
    'Groq Mixtral (Fast)': 'groq-mixtral',
    'Local Llama3 (Ollama)': 'llama3-local'
}

# Get model selection from sidebar or environment
selected_model_display = st.sidebar.selectbox(
    "Choose Language Model:",
    options=list(model_options.keys()),
    index=0,  # Default to GPT-4o
    help="Select the AI model for your job agent. Each model has different strengths and costs."
)

llm_name = model_options[selected_model_display]

# Override with environment variable if set
env_llm = os.environ.get('LLM_NAME')
if env_llm:
    llm_name = env_llm
    st.sidebar.info(f"Using model from environment: {env_llm}")

st.sidebar.markdown(f"**Selected Model:** `{llm_name}`")

# Model information
model_info = {
    'gemini-2.5-flash': "💫 Best price-performance with adaptive thinking capabilities",
    'gpt-4o': "🚀 Most capable model, excellent for complex reasoning",
    'gpt-4o-mini': "⚡ Fast and cost-effective, good for most tasks", 
    'openai': "🔄 Proven GPT-4 model, reliable performance",
    'claude-3.5-sonnet': "🧠 Excellent for analysis and reasoning tasks",
    'gemini-2.5-pro': "🔥 Google's most powerful thinking model with maximum accuracy",
    'gemini-2.5-flash-lite': "💰 Most cost-efficient model for high throughput tasks",
    'groq': "💨 Ultra-fast inference, good for real-time use",
    'groq-mixtral': "🔀 Fast mixture-of-experts model",
    'llama3-local': "🏠 Runs locally via Olloma, no API costs"
}

if llm_name in model_info:
    st.sidebar.markdown(f"ℹ️ {model_info[llm_name]}")

try:
    llm = load_llm(llm_name)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

uploaded_file = st.sidebar.file_uploader("Upload Your CV", type="pdf")
cv_text = "" # Initialize cv_text

print(f"Using model: {llm}")
st_callback = StreamlitCallbackHandler(st.container())
graph = define_graph(llm, llm_name)

# Handle file upload
if uploaded_file is not None:
    temp_dir = "tmp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    bytes_data = uploaded_file.getvalue()
    predefined_name = "cv.pdf"
    
    # To save the file, we use the 'with' statement to open a file and write the contents
    # The file will be saved with the predefined name in the /temp folder
    file_path = os.path.join(temp_dir, predefined_name)
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    
    # Load the CV content after saving it
    cv_text = load_cv(file_path)
    st.sidebar.success("✅ CV loaded and ready!")

    #@traceable # Auto-trace this function
    def conversational_chat(query, graph, cv_content):
        results = []
        # Gather job search filters from session state
        agent_state = {
            "messages": [HumanMessage(content=query)],
            "input": query,
            "cv_content": cv_content,
            "selected_cities": st.session_state.get("selected_cities", []),
            "remote_only": st.session_state.get("remote_only", False),
            "job_limit": st.session_state.get("job_limit", 10)
        }
        # Stream the response
        for s in graph.stream(
            agent_state,
            {"recursion_limit": 150}
        ):
            if "__end__" not in s:
                for key, value in s.items():
                    if key != "supervisor":
                        content_to_display = ""
                        agent_name = key
                        # Correctly parse the agent's output dictionary
                        if isinstance(value, dict) and "messages" in value and value["messages"]:
                            msg_obj = value["messages"][0]
                            content_to_display = getattr(msg_obj, 'content', str(msg_obj))
                            agent_name = getattr(msg_obj, 'name', key)
                            st.session_state.messages.append(AIMessage(content=content_to_display, name=agent_name))
                        else:
                            content_to_display = str(value)
                            st.session_state.messages.append(AIMessage(content=content_to_display))

                        print(f"DEBUG: Streaming output from node '{key}':\n{content_to_display}\n---")
                        
                        # Display the streaming message using st.markdown for better formatting
                        with st.chat_message(name=agent_name, avatar="🤖"):
                            st.markdown(content_to_display)
                        
                        results.append(content_to_display)

        st.session_state['history'].append((query, results))
        # Return the first result, which should be the complete response
        return results[0] if results else ""

    if 'selected_index' not in st.session_state:
        st.session_state['selected_index'] = None 
    # Initialize chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    # Initialize messages
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask anything to your Job agent: 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]

    # Create containers for chat history and user input
    response_container = st.container()
    container = st.container()

    # User input form
    with container:
        options =  [
                    "Extract and summarize my CV",
                    "Find me Data scientist job in Germany",
                    "Generate a cover letter for my cv",
                    "Find Data Scientist jobs in Germany, align them with my CV skills, and generate a cover letter tailored to my background."                
                    ]
        selected = pills(
                "Choose a question to get started or write your own below.",
                options,
                clearable=None, # type: ignore
                index=st.session_state['selected_index'],
                key="pills"
            )
        if selected:
            st.session_state['selected_index'] = options.index(selected)


        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", value=(selected if selected else st.session_state.get('input_text', '')), placeholder="Write your query 👉 (:", key='input')
            
            col1, col2 = st.columns(2)
            with col1:
                # Multi-select dropdown for popular German cities
                popular_cities = [
                    "Berlin", "Munich", "Hamburg", "Frankfurt", "Stuttgart", "Cologne", "Dusseldorf", "Leipzig", "Dresden", "Nuremberg", "Bremen", "Hanover", "Essen", "Dortmund", "Bonn"
                ]
                selected_cities = st.multiselect(
                    "Select cities (optional):",
                    options=popular_cities,
                    default=[],
                    help="Choose one or more cities to filter job search results."
                )
                # Checkbox for remote jobs
                remote_only = st.checkbox("Remote jobs only", value=False)
            
            with col2:
                # Number input for job limit
                job_limit = st.number_input(
                    "Number of jobs to fetch", 
                    min_value=1, 
                    max_value=100,  # Set a reasonable max to avoid long waits
                    value=10, 
                    help="Specify the maximum number of jobs to retrieve."
                )

            # Form submission buttons
            col3, col4 = st.columns(2)
            with col3:
                submit_button = st.form_submit_button(label='Send')
            with col4:
                export_button = st.form_submit_button(label='Search & Export to CSV')
            

        if submit_button and user_input:
            # Pass selected cities, remote status, and job limit to the agent state
            st.session_state['selected_cities'] = selected_cities
            st.session_state['remote_only'] = remote_only
            st.session_state['job_limit'] = job_limit
            output = conversational_chat(user_input, graph, cv_text)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['input_text'] = user_input  # Save the last input
            st.session_state['selected_index'] = None

        if export_button and user_input:
            st.session_state['selected_cities'] = selected_cities
            st.session_state['remote_only'] = remote_only
            st.session_state['job_limit'] = job_limit

            with st.spinner("Searching for jobs and generating CSV..."):
                # Run the conversational chat to get job results
                final_state = {}
                for s in graph.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "input": user_input,
                        "cv_content": cv_text,
                        "selected_cities": selected_cities,
                        "remote_only": remote_only,
                        "job_limit": job_limit,
                    },
                    {"recursion_limit": 150}
                ):
                    final_state.update(s)

                #print("DEBUG: final_state =", final_state)
                # Extract job_results from Searcher node if present
                job_data = []
                if "Searcher" in final_state and isinstance(final_state["Searcher"], dict):
                    job_data = final_state["Searcher"].get("job_results", [])
                else:
                    job_data = final_state.get("job_results", [])
                #print("DEBUG: job_data =", job_data)
                # Ensure job_data is a list of dicts, not a string
                import json
                if isinstance(job_data, str):
                    try:
                        job_data = json.loads(job_data)
                    except Exception as e:
                        st.error(f"Could not parse job data: {e}")
                        job_data = []

                # Store job data in session state
                if job_data:
                    st.session_state['job_data'] = job_data

    # Job selection UI - outside the export button block so it persists
    if 'job_data' in st.session_state and st.session_state['job_data']:
        st.markdown("---")
        st.markdown("### Job Selection")
        
        job_titles = [f"{job.get('job_title', 'Unknown')} at {job.get('company_name', 'Unknown')} ({job.get('job_location', 'Unknown')})" for job in st.session_state['job_data']]
        selected_job_idx = st.selectbox("Select a job to apply for:", options=list(range(len(job_titles))), format_func=lambda i: job_titles[i])
        selected_job = st.session_state['job_data'][selected_job_idx]
        
        st.markdown(f"### Selected Job\n**Title:** {selected_job.get('job_title', '')}\n**Company:** {selected_job.get('company_name', '')}\n**Location:** {selected_job.get('job_location', '')}")
        st.markdown(f"**Description:** {selected_job.get('job_desc_text', '')}")
        
        if st.button("Generate Improved CV & Cover Letter"):
            with st.spinner("Generating improved CV and cover letter using Generator agent..."):
                cover_letter, improved_cv = generate_cv_and_cover_letter(
                    llm, llm_name, cv_text, selected_job
                )
            st.markdown("#### Improved Cover Letter")
            st.markdown(cover_letter)
            if improved_cv:
                st.markdown("#### Improved CV")
                st.markdown(improved_cv)

        # CSV download section
        csv = convert_to_csv(st.session_state['job_data'])
        if csv:
            st.download_button(
                label="Download Jobs as CSV",
                data=csv,
                file_name="job_listings.csv",
                mime="text/csv",
            )
        else:
            st.error("Could not generate CSV. No job data found.")

    # Display chat history using st.chat_message and st.markdown
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                with st.chat_message(name="user", avatar="😊"):
                    st.markdown(st.session_state["past"][i])
                with st.chat_message(name="assistant", avatar="🤖"):
                    st.markdown(st.session_state["generated"][i])
