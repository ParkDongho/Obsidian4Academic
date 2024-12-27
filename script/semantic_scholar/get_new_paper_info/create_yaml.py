#!/usr/bin/env python3
import dotenv
dotenv.load_dotenv()

from typing import Any, Dict
import re

def create_yaml(metadata: Dict[str, Any], journal_dict, paper_id) -> Dict[str, Any]:
    authors = [author['name'] for author in metadata.get('authors', [])]
    title = metadata.get('title', 'Unknown Title')
    date = metadata.get('publicationDate', 'Unknown Date')
    year = metadata.get('year', 'Unknown Year')
    venue = metadata.get('venue', 'Unknown Venue')
    abstract = clean_abstract(metadata.get('abstract', 'No abstract available.'))
    citation_count = metadata.get('citationCount', 'Unknown')
    external_ids = metadata.get('externalIds', {})

    # external_ids에 새로운 키 추가
    external_ids['SEMANTIC'] = paper_id
    external_ids['IEEE'] = 1
    external_ids['ACM'] = 1

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

def clean_abstract(abstract: str) -> str:
    # "$\\time $"와 같은 패턴에서 닫는 $ 이전의 공백 제거
    return re.sub(r'(\\\S+)\s+\$', r'\1$', abstract)
