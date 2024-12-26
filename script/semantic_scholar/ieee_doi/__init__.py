from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def get_doi_from_ieee_paper(document_number, driver):
    try:
        # IEEE 문서 URL
        url = f"https://ieeexplore.ieee.org/document/{document_number}"
        driver.get(url)

        # DOI 정보 가져오기
        doi_element = driver.find_element(By.CSS_SELECTOR,
                                          "#xplMainContentLandmark > div > xpl-document-details > div > div.document-main.global-content-width-w-rr > div > div.document-main-content-container.col-19-24 > section > div.document-main-left-trail-content > div > xpl-document-abstract > section > div.abstract-desktop-div.hide-mobile.text-base-md-lh > div.row.g-0.u-pt-1 > div:nth-child(2) > div.u-pb-1.stats-document-abstract-doi > a"
                                          )
        doi_text = doi_element.accessible_name # DOI 텍스트 추출
        return doi_text
    except Exception as e:
        return f"오류 발생: {e}"


# 사용 예시
if __name__ == "__main__":
    document_number = "6091211"  # IEEE 논문 번호

    # Selenium 브라우저 설정
    service = Service(ChromeDriverManager().install())
    options = Options()
    # options.add_argument('--headless')  # 브라우저 창 숨기기
    options.add_argument('--disable-gpu')  # GPU 가속 비활성화
    options.add_argument('--no-sandbox')  # 권한 이슈 방지
    options.add_argument('--disable-dev-shm-usage')  # 메모리 부족 문제 해결

    driver = webdriver.Chrome(service=service, options=options)
    doi = get_doi_from_ieee_paper(document_number, driver)
    print(f"https://doi.org/{doi}")
