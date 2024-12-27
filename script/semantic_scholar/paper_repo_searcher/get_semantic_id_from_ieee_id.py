from script.semantic_scholar.paper_repo_searcher import get_doi_from_ieee_id, get_semantic_scholar_id_from_doi
from script.semantic_scholar.paper_repo_searcher.search_from_database import search_from_database

def get_semantic_id_from_ieee_id(ieee_paper_number, driver):
    """
    Get the Semantic Scholar ID using IEEE paper number.
    """
    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "IEEE", ieee_paper_number,
        "SEMANTIC", "/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/.paper_info")

    # Step 2: if not found, fetch DOI and Semantic Scholar ID
    if not semantic_id:
        doi = get_doi_from_ieee_id(ieee_paper_number, driver)
        semantic_id = get_semantic_scholar_id_from_doi(doi)
        # Step 2.1: Create new YAML file with this information

    return semantic_id

