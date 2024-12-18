#!/usr/bin/env python3
import dotenv
dotenv.load_dotenv()

import argparse
import time
import json
import os
import requests
from requests import Session
from typing import Generator, TypeVar

S2_API_KEY = os.environ.get('S2_API_KEY', '')

T = TypeVar('T')

# Tor 프록시 설정
proxies = {
  'http': 'socks5h://localhost:9050',
  'https': 'socks5h://localhost:9050',
}


def batched(items: list[T], batch_size: int) -> list[T]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def get_citation_batch(session: Session, paper_id: str, fields: str = 'paperId,title,venue,intents,isInfluential', retries = 3, backoff_factor=60.0, **kwargs) -> list[dict]:
    params = {
        'fields': fields,
        **kwargs,
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
                  "image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",  # Do Not Track 요청 헤더
        "Referer": "https://www.google.com/",
        'X-API-KEY': S2_API_KEY,
    }

    # Tor 네트워크를 통한 IP 확인 (선택 사항)
    try:
        ip_check = session.get("https://icanhazip.com", headers=headers)#, proxies=proxies)
        print(f"Using IP: {ip_check.text.strip()}")
    except requests.RequestException as e:
        print(f"Failed to check IP via Tor: {e}")

    url = f'https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references'
    for attempt in range(retries):
        try:
            with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                return response.json().get('data', [])
        except requests.exceptions.HTTPError as err:
            if response.status_code == 429:
                # Handle 429 error by waiting and retrying
                retry_after = int(response.headers.get("Retry-After", 60))  # Default to 60 seconds if not provided
                wait_time = backoff_factor * (2 ** attempt) + retry_after
                print(f"Rate limit hit. Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise the exception if it's not a 429 error
        except requests.RequestException as e:
            print(f"Request failed: {e}. Retrying...\nurl: {url}")
            time.sleep(backoff_factor * (2 ** attempt))

    # If all retries are exhausted
    raise requests.exceptions.HTTPError(f"Failed to fetch references after {retries} retries.\nurl: {url}")


def get_citations(paper_id: str, **kwargs) -> Generator[dict, None, None]:
    with Session() as session:
        yield from get_citation_batch(session, paper_id, **kwargs)


def main(args: argparse.Namespace) -> None:
    with open(args.s2id_file, 'r') as s2id_file:
        s2ids = [line.strip() for line in s2id_file.readlines()]

    fields = 'paperId,title,venue,intents,isInfluential'

    for paper_id in s2ids:
        citations = list(get_citations(paper_id, fields=fields))
        if not citations:
            print(f'No references found for paper ID {paper_id}')
            time.sleep(5)
            continue

        output_filename = f'{paper_id}-reference.json'
        with open(output_filename, 'w') as jsonfile:
            json.dump(citations, jsonfile, indent=4)

        print(f'Wrote references for paper ID {paper_id} to {output_filename}')
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', '-d', default='reference_json')
    parser.add_argument('s2id_file', nargs='?', default='../missing_reference.txt')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Change working directory to output directory
    os.chdir(args.output_dir)

    main(args)


