import requests
import re

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
