from dotenv import load_dotenv
load_dotenv()
from llms import load_llm

print('Testing Gemini models with LangSmith tracing enabled...')
gemini_models = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']
for model in gemini_models:
    print(f'--- {model} ---')
    try:
        llm = load_llm(model)
        print(f'✓ {model}: Successfully loaded')
        print(f'  Model: {llm.model_name}')
        print(f'  API Key configured: {bool(getattr(llm, "google_api_key", None))}')
    except Exception as e:
        print(f'✗ {model}: Error - {e}')
