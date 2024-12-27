import os
from requests import Session
import time
import yaml
import pandas as pd
from typing import Any, Dict
import re
import dotenv
import requests

dotenv.load_dotenv()

PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
JOURNAL_LIST_PATH = os.environ.get('JOURNAL_LIST_PATH', '')
S2_API_KEY = os.environ.get('S2_API_KEY', '')


def get_redirected_url(doi):
    """
    DOI를 사용해 최종 리다이렉션된 URL을 반환합니다.
    """
    base_url = "https://doi.org/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(base_url + doi, allow_redirects=True, timeout=10, headers=headers)
        response.raise_for_status()
        return response.url
    except requests.exceptions.RequestException as e:
        print(f"Error accessing DOI: {e}")
        return None


def identify_source_and_id(url):
    """
    URL에서 출처와 DOI 또는 문서 ID를 추출합니다.
    """
    if "ieee.org" in url:
        source = "IEEE"
        match = re.search(r"/document/(\d+)", url)
        document_id = match.group(1) if match else "Unknown"
    elif "dl.acm.org" in url:
        source = "ACM"
        match = re.search(r"/doi/(10\.\d{4,}/.+)", url)
        document_id = match.group(1) if match else "Unknown"
    else:
        source = "Other"
        document_id = "Unknown"
    return source, document_id

def get_journal_id_from_doi(doi):
    """
    사용자로부터 DOI 입력을 받고 결과를 출력합니다.
    """
    redirected_url = get_redirected_url(doi)
    if redirected_url:
        source, document_id = identify_source_and_id(redirected_url)
        return source, document_id
    else:
        print("Failed to retrieve URL.")
        return None, None

def create_yaml(metadata: Dict[str, Any], paper_id) -> Dict[str, Any]:
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

def download_paper_info(semantic_id: str) -> None:
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

def save_paper_info(s2id_file: str) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(PAPER_INFO_PATH, exist_ok=True)

    # Change working directory to output directory
    os.chdir(PAPER_INFO_PATH)

    get_paper_info(s2id_file)

if __name__ == "__main__":
    save_paper_info(
        '/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/new_paper_list.txt',
    )
