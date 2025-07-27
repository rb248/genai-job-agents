import os
import nest_asyncio
nest_asyncio.apply()
from linkedin_api import Linkedin

api = Linkedin(os.environ["LINKEDIN_EMAIL"], os.environ["LINKEDIN_PASS"])

def get_job_type(job_type):
    job_type_mapping = {
        "full-time": "F",
        "contract": "C",
        "part-time": "P",
        "temporary": "T",
        "internship": "I",
        "volunteer": "V",
        "other": "O"
    }

    return job_type_mapping.get(job_type.lower())

def get_job_ids(keywords, location_name, job_type=None, limit=10, companies=None, industries=None, remote=None):
    if job_type is not None:
        job_type = get_job_type(job_type)
    print(f"DEBUG: get_job_ids called with keywords={keywords}, location_name={location_name}, job_type={job_type}, limit={limit}, companies={companies}, industries={industries}, remote={remote}")

    # Support multiple locations
    if isinstance(location_name, str):
        location_names = [location_name]
    else:
        location_names = location_name if location_name else []
    job_ids = []
    for loc in location_names:
        print(f"DEBUG: api.search_jobs called with location_name={loc}, remote={remote}")
        job_postings = api.search_jobs(
            keywords=keywords,
            job_type=job_type,
            location_name=loc,
            companies=companies,
            industries=industries,
            remote=remote,
            limit=limit
        )
        print(f"DEBUG: api.search_jobs returned {len(job_postings)} jobs for location_name={loc}")
        job_ids += [job['trackingUrn'].split('jobPosting:')[1] for job in job_postings]
    return job_ids

import nest_asyncio
nest_asyncio.apply()

async def get_job_details(job_id):
    try:
        job_data = api.get_job(job_id)  # Assuming this function is async and fetches job data
        #print(f"DEBUG: Raw job_data for job_id {job_id}: {job_data}")
        # Construct the job data dictionary with defaults
        job_data_dict = {
            "company_name": job_data.get('companyDetails', {}).get('com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany', {}).get('companyResolutionResult', {}).get('name', ''),
            "company_url": job_data.get('companyDetails', {}).get('com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany', {}).get('companyResolutionResult', {}).get('url', ''),
            "job_desc_text": job_data.get('description', {}).get('text', ''),
            "work_remote_allowed": job_data.get('workRemoteAllowed', ''),
            "job_title": job_data.get('title', ''),
            "company_apply_url": job_data.get('applyMethod', {}).get('com.linkedin.voyager.jobs.OffsiteApply', {}).get('companyApplyUrl', ''),
            "job_location": job_data.get('formattedLocation', '')
        }
        #print(f"DEBUG: Parsed job_data_dict for job_id {job_id}: {job_data_dict}")
    except Exception as e:
        print(f"Error fetching job details for job ID {job_id}: {str(e)}")
        job_data_dict = {
            "company_name": '',
            "company_url": '',
            "job_desc_text": '',
            "work_remote_allowed": '',
            "job_title": '',
            "company_apply_url": '',
            "job_location": ''
        }
    return job_data_dict

async def fetch_all_jobs(job_ids, batch_size=10):
    results = []
    for job_id in job_ids:
        job_detail = await get_job_details(job_id)
        results.append(job_detail)
    return results

async def job_threads(job_ids):
    return await fetch_all_jobs(job_ids, 10)