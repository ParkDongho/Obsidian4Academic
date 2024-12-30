import os
from requests import Session
import time
import yaml
import pandas as pd
from typing import Any, Dict
import re
import dotenv

from academic_copilot.semantic_scholar.get_journal_id import get_journal_id_from_doi

dotenv.load_dotenv()

PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
JOURNAL_LIST_PATH = os.environ.get('JOURNAL_LIST_PATH', '')
S2_API_KEY = os.environ.get('S2_API_KEY', '')
NEW_PAPER_LIST = os.environ.get('NEW_PAPER_LIST', '')

def create_yaml(metadata, paper_id):
    authors = [author['name'] for author in metadata.get('authors', [])]
    title = metadata.get('title', 'Unknown Title')
    date = metadata.get('publicationDate', 'Unknown Date')
    year = metadata.get('year', 'Unknown Year')
    venue = metadata.get('venue', 'Unknown Venue')
    abstract = clean_abstract(metadata.get('abstract', 'No abstract available.'))
    citation_count = metadata.get('citationCount', 'Unknown')
    external_ids = metadata.get('externalIds', {})

    journal_list = load_journal_list(JOURNAL_LIST_PATH)

    # external_ids에 새로운 키 추가
    external_ids['SEMANTIC'] = paper_id

    external_ids['IEEE'] = None
    external_ids['ACM'] = None
    doi = external_ids.get('DOI', None)
    if doi != None:
        journal_key = get_journal_id_from_doi(doi)
        if journal_key[0] == "IEEE":
            external_ids['IEEE'] = journal_key[1]
        elif journal_key[0] == "ACM" in doi:
            external_ids['ACM'] = journal_key[1]


    short_name = "Unknown Venue"
    for key in journal_list:
        if key.lower() in venue.lower():
            short_name = journal_list[key]
            break

    yaml_data = {
        'title': title,
        'date': date,
        'authors': authors,
        'year': year,
        'venue': venue,
        'venue_short': short_name,
        'abstract': abstract,
        'citation_count': citation_count,
        'external_ids': external_ids
    }
    return yaml_data

def clean_abstract(abstract: str) -> str:
    # "$\\time $"와 같은 패턴에서 닫는 $ 이전의 공백 제거
    return re.sub(r'(\\\S+)\s+\$', r'\1$', abstract)



def get_paper_metadata(session: Session, paper_id: str,
                       fields: str = 'title,authors,year,venue,abstract,citationCount') -> Dict[str, Any]:
    params = {
        'fields': fields,
    }
    headers = {
        'X-API-KEY': S2_API_KEY,
    }

    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}'
    with session.get(url, params=params, headers=headers) as response:
        response.raise_for_status()
        return response.json()


def load_journal_list(csv_file_path):
    """
    Journal list CSV 파일을 읽어 딕셔너리로 반환
    :param csv_file_path:
    :return: jounal_list_dict
    """
    # CSV 파일을 읽어 'journal'과 'short' 컬럼을 딕셔너리로 반환
    df = pd.read_csv(csv_file_path)
    return dict(zip(df['journal'], df['name_short']))

def download_paper_info(semantic_id):
    # Change working directory to output directory
    PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
    os.chdir(PAPER_INFO_PATH)

    fields = 'title,authors,year,venue,abstract,citationCount,externalIds,publicationDate'
    with Session() as session:
        paper_metadata = get_paper_metadata(session, semantic_id, fields=fields)

    if not paper_metadata:
        print(f'No metadata found for paper ID {semantic_id}')
        return None

    yaml_content = create_yaml(paper_metadata, semantic_id)
    output_filename = f'{semantic_id}.yaml'
    with open(output_filename, 'w') as yamlfile:
        yaml.dump(yaml_content, yamlfile, default_flow_style=False, allow_unicode=True)

    time.sleep(3)
    print(f'Wrote YAML for paper ID {semantic_id} to {output_filename}')



def get_paper_info(s2id_file):
    """
    Get paper metadata from Semantic Scholar API and save as YAML files.
    :param s2id_file: paper_list 가 있는 파일 경로
    :return:
    """
    # Change working directory to output directory
    PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
    os.chdir(PAPER_INFO_PATH)

    with open(s2id_file, 'r') as s2id_file:
        s2ids = [line.strip() for line in s2id_file.readlines()]
    fields = 'title,authors,year,venue,abstract,citationCount,externalIds,publicationDate'
    for paper_id in s2ids:
        with Session() as session:
            paper_metadata = get_paper_metadata(session, paper_id, fields=fields)

        if not paper_metadata:
            print(f'No metadata found for paper ID {paper_id}')
            continue

        yaml_content = create_yaml(paper_metadata, paper_id)
        output_filename = f'{paper_id}.yaml'
        with open(output_filename, 'w') as yamlfile:
            yaml.dump(yaml_content, yamlfile, default_flow_style=False, allow_unicode=True)

        time.sleep(3)
        print(f'Wrote YAML for paper ID {paper_id} to {output_filename}')




def save_paper_info(s2id_file):
    PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
    # Create output directory if it doesn't exist
    os.makedirs(PAPER_INFO_PATH, exist_ok=True)

    get_paper_info(s2id_file)

def save_paper_info_from_id(semantic_id):
    PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
    os.makedirs(PAPER_INFO_PATH, exist_ok=True)
    download_paper_info(semantic_id)

if __name__ == "__main__":
    save_paper_info(NEW_PAPER_LIST)
