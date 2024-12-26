import os
import yaml
import requests
import semantic.ieee_doi

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def get_semantic_scholar_id_from_doi(doi):
    """
    Get the Semantic Scholar ID using DOI.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=paperId"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('paperId', None)
    return None





def map_ieee_to_semantic(directory, ieee_paper_number):
    """
    Map IEEE paper number to Semantic Scholar ID, checking YAML files in a directory.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".yaml"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                content = yaml.safe_load(file)
                if 'external_ids' in content and content['external_ids'].get('IEEE') == ieee_paper_number:
                    return content['external_ids'].get('CorpusId')

    # If not found, fetch DOI and Semantic Scholar ID
    doi = get_doi_from_ieee_paper(ieee_paper_number)
    semantic_id = get_semantic_scholar_id_from_doi(doi)

    if semantic_id:
        # Create new YAML file with this information
        new_file_content = {
            "ieee_paper_number": ieee_paper_number,
            "external_ids": {
                "DOI": doi,
                "CorpusId": semantic_id
            }
        }
        new_file_path = os.path.join(directory, f"{ieee_paper_number}.yaml")
        with open(new_file_path, 'w') as new_file:
            yaml.dump(new_file_content, new_file)
        return semantic_id

    return None


# Example usage:
directory = "/path/to/your/yaml/files"  # Replace with your directory path
ieee_paper_number = "3007787.3001177"  # Replace with your IEEE paper number
semantic_id = map_ieee_to_semantic(directory, ieee_paper_number)

if semantic_id:
    print(f"Semantic Scholar ID for IEEE paper {ieee_paper_number}: {semantic_id}")
else:
    print(f"Semantic Scholar ID for IEEE paper {ieee_paper_number} could not be found.")
