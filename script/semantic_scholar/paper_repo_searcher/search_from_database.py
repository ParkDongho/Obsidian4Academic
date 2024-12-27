import os
import yaml

def search_from_database(key, value, result_key, database_path):
    """
    Search for a key-value pair in the YAML files.
    """
    for filename in os.listdir(database_path):
        if filename.endswith(".yaml"):
            file_path = os.path.join(database_path, filename)
            with open(file_path, 'r') as file:
                content = yaml.safe_load(file)
                if 'external_ids' in content and content['external_ids'].get(key) == value:
                    return content['external_ids'].get(result_key)
    return None


if __name__ == "__main__":
    key = "IEEE"
    value = "8686088"
    result_key = "SEMANTIC"
    database_path = "/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/.paper_info"
    semantic_id = search_from_database(key, value, result_key, database_path)

    if semantic_id:
        print(f"Semantic Scholar ID for {key} {value}: {semantic_id}")
    else:
        print(f"Semantic Scholar ID for {key} {value}: not found.")
