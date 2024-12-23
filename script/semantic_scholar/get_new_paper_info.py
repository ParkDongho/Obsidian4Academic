#!/usr/bin/env python3
import dotenv
dotenv.load_dotenv()

import argparse
import os
import json
from requests import Session
from typing import Any, Dict
import pandas as pd
import time
import yaml
import re

S2_API_KEY = os.environ.get('S2_API_KEY', '')

def load_journal_dict(csv_file_path):
    # CSV 파일을 읽어 'journal'과 'short' 컬럼을 딕셔너리로 반환
    df = pd.read_csv(csv_file_path)
    return dict(zip(df['journal'], df['name_short']))

def get_paper_metadata(session: Session, paper_id: str, fields: str = 'title,authors,year,venue,abstract,citationCount') -> Dict[str, Any]:
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

def clean_abstract(abstract: str) -> str:
    # "$\\time $"와 같은 패턴에서 닫는 $ 이전의 공백 제거
    return re.sub(r'(\\\S+)\s+\$', r'\1$', abstract)

def create_yaml(metadata: Dict[str, Any], journal_dict) -> Dict[str, Any]:
    authors = [author['name'] for author in metadata.get('authors', [])]
    title = metadata.get('title', 'Unknown Title')
    date = metadata.get('publicationDate', 'Unknown Date')
    year = metadata.get('year', 'Unknown Year')
    venue = metadata.get('venue', 'Unknown Venue')
    abstract = clean_abstract(metadata.get('abstract', 'No abstract available.'))
    citation_count = metadata.get('citationCount', 'Unknown')
    external_ids = metadata.get('externalIds', {})

    short_name = "Unknown Venue"
    for key in journal_dict:
        if key.lower() in venue.lower():
            short_name = journal_dict[key]
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

def main(args: argparse.Namespace, journal_dict) -> None:
    with open(args.s2id_file, 'r') as s2id_file:
        s2ids = [line.strip() for line in s2id_file.readlines()]

    fields = 'title,authors,year,venue,abstract,citationCount,externalIds,publicationDate'
    for paper_id in s2ids:
        with Session() as session:
            paper_metadata = get_paper_metadata(session, paper_id, fields=fields)
        
        if not paper_metadata:
            print(f'No metadata found for paper ID {paper_id}')
            continue

        yaml_content = create_yaml(paper_metadata, journal_dict)
        output_filename = f'{paper_id}.yaml'
        with open(output_filename, 'w') as yamlfile:
            yaml.dump(yaml_content, yamlfile, default_flow_style=False, allow_unicode=True)

        time.sleep(3)
        print(f'Wrote YAML for paper ID {paper_id} to {output_filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output-dir', '-d',
        default='/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/.paper_info/',
        help="Directory to save output files"
    )
    parser.add_argument(
        '--s2id-file',
        default='/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/new_paper_list.txt',
        help="Path to the S2ID file"
    )
    parser.add_argument(
        '--csv-file',
        default='/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/journal_list.csv',
        help="Path to the journal list CSV file"
    )

    args = parser.parse_args()
    
    # Load the journal dictionary from the CSV file
    journal_dict = load_journal_dict(args.csv_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory to output directory
    os.chdir(args.output_dir)

    main(args, journal_dict)

