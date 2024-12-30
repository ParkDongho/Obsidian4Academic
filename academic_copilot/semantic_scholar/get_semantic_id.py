import requests

from academic_copilot.semantic_scholar.get_doi_from_ieee_id import get_doi_from_ieee_id
from academic_copilot.semantic_scholar.search_from_database import search_from_database
from academic_copilot.semantic_scholar.get_paper_info import save_paper_info, save_paper_info_from_semantic_id

import os
PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')

def get_semantic_id_from_doi(doi_id, ieee_paper_id=None, acm_paper_id=None):
    """
    Get the Semantic **Scholar ID** using **DOI**. `(DOI -> Semantic Scholar ID)`

    - Step 1: search for the DOI number in the YAML files
    - Step 2: if not found, fetch Semantic Scholar ID from DOI
    - Step 2.1: Create new YAML file with this information

    :param doi_id: DOI number
    :param ieee_paper_id: IEEE paper number
    :param acm_paper_id: ACM paper number
    :returns: Semantic Scholar ID
    """

    # Step 1: search for the DOI number in the YAML files
    semantic_id = search_from_database(
        "DOI", doi_id,
        "SEMANTIC")

    # Step 2: if not found, fetch Semantic Scholar ID from DOI
    if not semantic_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_id}?fields=paperId"
        response = requests.get(url)

        if response.status_code == 200:
            semantic_id = response.json().get('paperId', None)
            # Step 2.1: Create new YAML file with this information
            save_paper_info_from_semantic_id(semantic_id,
                                             ieee_paper_id=ieee_paper_id, acm_paper_id=acm_paper_id, doi_id=doi)

            # return : Step 2의 결과가 있을 경우
            return semantic_id

        # return : Step 2의 결과가 없을 경우
        return None

    # return : Step 1(database 검색)의 결과가 있을 경우
    return semantic_id


def get_semantic_id_from_ieee_id(ieee_paper_id, driver, acm_paper_id=None):
    """
    Get the Semantic Scholar ID using IEEE paper number.

    - **Step 1:** search for the IEEE paper number in the YAML files
    - **Step 2:** if not found, fetch DOI and Semantic Scholar ID
    - Step 2.1: fetch semantic scholar id from DOI

    :param ieee_paper_id: IEEE paper number
    :param driver: Selenium WebDriver
    :param acm_paper_id: ACM paper number
    :returns: Semantic Scholar ID
    """

    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "IEEE", ieee_paper_id,
        "SEMANTIC")

    # Step 2: if not found, fetch DOI from ieee_id
    if not semantic_id:
        tmp_doi = get_doi_from_ieee_id(ieee_paper_id, driver)

        # Step 2.1: fetch semantic scholar id from DOI
        return get_semantic_id_from_doi(tmp_doi, ieee_paper_id=ieee_paper_id, acm_paper_id=acm_paper_id)

    return semantic_id



if __name__ == "__main__":
    doi = "10.1109/JSSC.2016.2616357"
    semantic_id = get_semantic_id_from_doi(doi)
    if semantic_id:
        print(f"Semantic Scholar ID for DOI {doi}: {semantic_id}")
    else:
        print(f"Semantic Scholar ID for DOI {doi} could not be found.")

