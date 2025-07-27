import pandas as pd
import streamlit as st

def convert_to_csv(job_data):
    """Converts a list of job dictionaries to a CSV string."""
    if not job_data or not isinstance(job_data, list):
        return None
    
    # Ensure all dictionaries have the same keys
    all_keys = set()
    for item in job_data:
        if isinstance(item, dict):
            all_keys.update(item.keys())

    # Standardize dictionaries
    standardized_data = []
    for item in job_data:
        if isinstance(item, dict):
            standardized_item = {key: item.get(key, "") for key in all_keys}
            standardized_data.append(standardized_item)

    if not standardized_data:
        return None

    df = pd.DataFrame(standardized_data)
    return df.to_csv(index=False).encode('utf-8')

def get_job_details_for_export(results):
    """Extracts job details from agent results for CSV export."""
    # This is a placeholder. The actual implementation will depend on
    # how the job data is structured in the agent's output.
    # For now, we assume the 'Searcher' provides a list of job dicts.
    for result in results:
        if isinstance(result, dict) and result.get("name") == "Searcher":
            # Assuming the content is a list of job dictionaries
            return result.get("content", [])
    return []
