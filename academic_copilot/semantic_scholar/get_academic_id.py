import requests
import re
import os

from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager

from academic_copilot.semantic_scholar.search_from_database import search_from_database
from academic_copilot.semantic_scholar.get_paper_info import save_paper_info_from_semantic_id

PAPER_INFO_PATH = os.environ.get('PAPER_INFO_PATH', '')
S2_API_KEY = os.environ.get('S2_API_KEY', '')
CITATION_INFO_PATH = os.environ.get('CITATION_INFO_PATH', '')

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

def get_semantic_id_from_doi(doi_id, ieee_paper_id=None, acm_paper_id=None):
    """
    Get the Semantic **Scholar ID** using **DOI**. `(DOI -> Semantic Scholar ID)`

    - Step 1: search for the DOI number in the YAML files
    - Step 2: if not found, fetch Semantic Scholar ID from DOI
    - Step 2.1: Create new YAML file with this information

    :param doi_id: DOI number
    :param ieee_paper_id: IEEE paper number
    :param acm_paper_id: ACM paper number
    :returns: Semantic Scholar ID
    """

    # Step 1: search for the DOI number in the YAML files
    semantic_id = search_from_database(
        "DOI", doi_id,
        "SEMANTIC")

    # Step 2: if not found, fetch Semantic Scholar ID from DOI
    if not semantic_id:
        url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_id}?fields=paperId"
        response = requests.get(url)

        if response.status_code == 200:
            semantic_id = response.json().get('paperId', None)
            # Step 2.1: Create new YAML file with this information
            save_paper_info_from_semantic_id(semantic_id,
                                             ieee_paper_id=ieee_paper_id, acm_paper_id=acm_paper_id, doi_id=doi_id)

            # return : Step 2의 결과가 있을 경우
            return semantic_id

        # return : Step 2의 결과가 없을 경우
        return None

    # return : Step 1(database 검색)의 결과가 있을 경우
    return semantic_id


def get_semantic_id_from_ieee_id(ieee_paper_id, driver, acm_paper_id=None):
    """
    Get the Semantic Scholar ID using IEEE paper number.

    - **Step 1:** search for the IEEE paper number in the YAML files
    - **Step 2:** if not found, fetch DOI and Semantic Scholar ID
    - Step 2.1: fetch semantic scholar id from DOI

    :param ieee_paper_id: IEEE paper number
    :param driver: Selenium WebDriver
    :param acm_paper_id: ACM paper number
    :returns: Semantic Scholar ID
    """

    # Step 1: search for the IEEE paper number in the YAML files
    semantic_id = search_from_database(
        "IEEE", ieee_paper_id,
        "SEMANTIC")

    # Step 2: if not found, fetch DOI from ieee_id
    if not semantic_id:
        tmp_doi = get_doi_from_ieee_id(ieee_paper_id, driver)

        # Step 2.1: fetch semantic scholar id from DOI
        return get_semantic_id_from_doi(tmp_doi, ieee_paper_id=ieee_paper_id, acm_paper_id=acm_paper_id)

    return semantic_id

def get_doi_from_ieee_id(ieee_number, driver):
    try:
        # IEEE 문서 URL
        url = f"https://ieeexplore.ieee.org/document/{ieee_number}"
        driver.get(url)

        # DOI 정보 가져오기
        doi_element = driver.find_element(By.CSS_SELECTOR,
            "#xplMainContentLandmark > div > xpl-document-details > div > div.document-main.global-content-width-w-rr > "
            "div > div.document-main-content-container.col-19-24 > section > div.document-main-left-trail-content > "
            "div > xpl-document-abstract > section > div.abstract-desktop-div.hide-mobile.text-base-md-lh > "
            "div.row.g-0.u-pt-1 > div:nth-child(2) > div.u-pb-1.stats-document-abstract-doi > a")

        doi_text = doi_element.accessible_name # DOI 텍스트 추출
        return doi_text

    except Exception as e:
        return f"오류 발생: {e}"
