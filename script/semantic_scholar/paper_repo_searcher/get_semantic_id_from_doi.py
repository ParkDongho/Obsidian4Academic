import requests
from script.semantic_scholar.paper_repo_searcher.search_from_database import search_from_database

import os
PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')

def get_semantic_scholar_id_from_doi(doi):
    """
    Get the Semantic Scholar ID using DOI.
    """
    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "DOI", doi,
        "SEMANTIC", PAPER_INFO_PATH)

    if not semantic_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=paperId"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('paperId', None)
        return None
    return semantic_id


if __name__ == "__main__":
    doi = "10.1109/JSSC.2016.2616357"
    semantic_id = get_semantic_scholar_id_from_doi(doi)
    if semantic_id:
        print(f"Semantic Scholar ID for DOI {doi}: {semantic_id}")
    else:
        print(f"Semantic Scholar ID for DOI {doi} could not be found.")

