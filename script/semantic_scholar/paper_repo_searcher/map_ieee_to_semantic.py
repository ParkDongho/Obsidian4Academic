import os
import yaml
import get_doi_from_ieee_id
from script.semantic_scholar.paper_repo_searcher import get_semantic_scholar_id_from_doi


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
    doi = get_doi_from_ieee_id(ieee_paper_number)
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


