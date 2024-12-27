import requests
from semantic_scholar import get_doi_from_ieee_id
from semantic_scholar.get_semantic_id_from_doi import get_semantic_scholar_id_from_doi
from semantic_scholar.search_from_database import search_from_database
from semantic_scholar.get_paper_info import save_paper_info

import os
PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')

def get_semantic_scholar_id_from_doi(doi_id):
    """
    Get the Semantic Scholar ID using DOI.
    """
    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "DOI", doi_id,
        "SEMANTIC", PAPER_INFO_PATH)

    if not semantic_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_id}?fields=paperId"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('paperId', None)
        return None
    return semantic_id


def get_semantic_id_from_ieee_id(ieee_paper_number, driver):
    """
    Get the Semantic Scholar ID using IEEE paper number.
    """
    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "IEEE", ieee_paper_number,
        "SEMANTIC", PAPER_INFO_PATH)

    # Step 2: if not found, fetch DOI and Semantic Scholar ID
    if not semantic_id:
        doi = get_doi_from_ieee_id(ieee_paper_number, driver)
        semantic_id = get_semantic_scholar_id_from_doi(doi)
        # Step 2.1: Create new YAML file with this information
        save_paper_info(semantic_id)

    return semantic_id



if __name__ == "__main__":
    doi = "10.1109/JSSC.2016.2616357"
    semantic_id = get_semantic_scholar_id_from_doi(doi)
    if semantic_id:
        print(f"Semantic Scholar ID for DOI {doi}: {semantic_id}")
    else:
        print(f"Semantic Scholar ID for DOI {doi} could not be found.")

