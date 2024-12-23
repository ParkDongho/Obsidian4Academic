from bs4 import BeautifulSoup

def extract_references_from_html(file_path):
    """
    HTML 파일에서 참조와 링크를 추출합니다.

    Args:
        file_path (str): 로컬 HTML 파일 경로

    Returns:
        list: 참조 번호, 제목, 링크 정보가 포함된 딕셔너리 리스트
    """
    # HTML 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    # 참조 목록 추출
    reference_list = []
    references = soup.select(".reference-container")

    for ref in references:
        try:
            # 참조 번호
            number_element = ref.select_one(".number b")
            number = number_element.text.strip() if number_element else None

            # 제목
            title_element = ref.select_one(".col > div:first-child")
            title = title_element.text.strip() if title_element else None

            # 링크들
            links = {}
            link_elements = ref.select(".ref-link a")
            for link in link_elements:
                link_text = link.text.strip()
                href = link.get("href")
                if link_text and href:
                    links[link_text] = href

            reference_list.append({
                "number": number,
                "title": title,
                "links": links
            })
        except Exception as e:
            print(f"Error processing reference: {e}")

    return reference_list

# 실행 예제
if __name__ == "__main__":
    input_file = "./test/reference.html"  # 입력 HTML 파일 경로
    references = extract_references_from_html(input_file)

    # 결과 출력
    for ref in references:
        print(f"Reference {ref['number']}: {ref['title']}")
        for link_text, href in ref['links'].items():
            print(f"  - {link_text}: {href}")
        print()
