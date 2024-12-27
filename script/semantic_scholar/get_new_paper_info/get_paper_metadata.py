import os
from requests import Session
from typing import Any, Dict

S2_API_KEY = os.environ.get('S2_API_KEY', '')


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

