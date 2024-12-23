from bs4 import BeautifulSoup
import argparse
import os
import json
import requests
import argparse

from orca.punctuation_settings import section
from selenium import webdriver
import time

# Load the HTML file
fig_table_data = []

from bs4 import Tag, NavigableString

def parsePaper(input, ieee_paper_info):
    sections = []
    for section in input:
        single_section = parseSection(section, ieee_paper_info)
        sections = sections + single_section
    return sections


def parseSection(input, ieee_paper_info):

    single_section = []
    subsec_list = []
    heading_level = None
    section_title = ""
    section_content = ""
    section_id = ""

    for paragraph in input.contents:
        # Tag인지 NavigableString인지 확인
        if isinstance(paragraph, NavigableString):
            if (paragraph.strip() != ""):
                print(f"Invalid NavigableString!: {paragraph}")
            continue

        # subsection 처리
        if 'section_2' in paragraph.get('class', "div"):
            subsec_list = subsec_list + parseSection(paragraph, ieee_paper_info)

        else:
            if (paragraph.name in ["h3", "h4", "h5", "h6"]) or ('header' in paragraph.get('class', "div")):
                for level in range(2, 7):
                    heading_tag = input.find(f'h{level}')
                    if heading_tag:
                        heading_level = level
                        section_title = heading_tag.text.strip()
                        section_id = input.attrs['id']
                        break

            elif paragraph.name == "disp-formula":
                # 찾고자 하는 <span> 태그와 그 내부의 텍스트를 추출
                span_tag = paragraph.find('span', class_="tex tex2jax_ignore")
                if span_tag and span_tag.text:
                    latex_code = span_tag.text.strip()  # 텍스트를 가져와서 공백 제거
                    section_content += f"\n$$\n{latex_code}\n$$\n\n"
                else:
                    print("No latex code found in disp-formula")

            # 본문 처리
            elif paragraph.name == "p":

                # inline-formula 처리
                for math in paragraph.find_all('inline-formula'):
                    latex_code = math.find('script', {'type': 'math/tex'}).text
                    math.replace_with(f"${latex_code}$")


                for math in paragraph.find_all('disp-formula'):
                    span_tag = math.find('span', class_="tex tex2jax_ignore")
                    if span_tag and span_tag.text:
                        latex_code = span_tag.text.strip()  # 텍스트를 가져와서 공백 제거
                        math.replace_with(f"\n$$\n{latex_code}\n$$\n\n")
                    else:
                        print("No latex code found in disp-formula")

                # 참조 처리
                for link in paragraph.find_all('a'):
                    # bibriography 처리
                    if link.attrs['ref-type'] == "bibr":
                        link_text = link.text
                        if link.text[0] == "[":
                            link_text = link.text.replace("[", "\[")
                        if link.text[-1] == "]":
                            link_text = link_text.replace("]", "\]")
                        link.replace_with(f"[{link_text}]({link.attrs['anchor']})")

                    # 본문 내 이미지 관련 참조 처리
                    elif link.attrs['ref-type'] == "fig":
                        link.replace_with(f"[{link.text}]({link.attrs['anchor']})")

                    # 본문 내 표 관련 참조 처리
                    elif link.attrs['ref-type'] == "table":
                        link.replace_with(f"[{link.text}]({link.attrs['anchor']})")

                    # 본문 내 섹션 관련 참조 처리
                    elif link.attrs['ref-type'] == "sec":
                        link.replace_with(f"[{link.text}]({link.attrs['anchor']})")

                    # 본문 내 수식 관련 참조 처리
                    elif link.attrs['ref-type'] == "fn":
                        link.replace_with(f"[{link.text}]({link.attrs['anchor']})")

                    else:
                        print("Unhandled Link: ", link.text, ", ", link.attrs['ref-type'], ", ", link.attrs['anchor'])

                paragraph_text = ' '.join(paragraph.stripped_strings)
                section_content += paragraph_text + "\n\n"



            # Figure 생성
            elif any(cls in paragraph.get('class', []) for cls in ["figure", "figure-full", "table"]):
                # 이미지와 캡션 파싱
                image_wrap = paragraph.find('div', class_='img-wrap')
                fig_caption = paragraph.find('div', class_='figcaption')

                if image_wrap and fig_caption:
                    # 링크와 이미지 경로 추출
                    link_tag = image_wrap.find('a')
                    img_tag = link_tag.find('img') if link_tag else None
                    image_href = link_tag['href'] if link_tag else ''
                    alt_text = img_tag.get('alt', 'Image') if img_tag else ''
                    data_fig_id = paragraph.get('id') if img_tag else ''

                    # 캡션 추출
                    caption_title = fig_caption.find('b', class_='title').get_text(strip=True) \
                        if fig_caption.find('b', class_='title') else ''
                    caption_text = fig_caption.find('p').get_text(strip=True) if fig_caption.find('p') else ''

                    img_file_name = f"ieee_{ieee_paper_info['ieee_paper_id']}_{data_fig_id}.gif"
                    img_file_path = f"{ieee_paper_info['relative_img_dir']}/{img_file_name}"

                    # 마크다운 형식으로 변환
                    markdown_output = f"![{alt_text}]({img_file_path})\n\n**{caption_title}** {caption_text}"

                    # 섹션 내용에 추가
                    section_content += f"\n{markdown_output}\n"
                    fig_table_data.append({
                        "image_href": f"https://ieeexplore.ieee.org/{image_href}",
                        "img_file_name": img_file_name,
                        "data_fig_id": data_fig_id
                    })

                else:
                    section_content += "\nFigure could not be parsed.\n"


            else:
                print("\n\nUnhandled Tag:", paragraph)


        single_section = [(heading_level, section_title, section_content, section_id)] + subsec_list

    return single_section

# Function to extract content and convert to Markdown
def html_to_markdown(driver, ieee_paper_info):
    # with open(html_path, 'r', encoding='utf-8') as file:
    #     html_content = file.read()

    driver.get(f"https://ieeexplore.ieee.org/document/{ieee_paper_info['ieee_paper_id']}")

    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Extract title
    title = soup.find('title').text.strip() if soup.find('title') else "No Title"

    # Extract abstract
    abstract_section = soup.find('meta', {'name': 'Description'})
    abstract = abstract_section['content'] if abstract_section else "No Abstract"

    # Extract authors
    authors = [tag['content'] for tag in soup.find_all('meta', {'name': 'parsely-author'})]

    # Extract sections and process paragraphs
    sections = parsePaper(soup.find_all('div', class_=['section']), ieee_paper_info)

    # Convert to Markdown
    markdown_content = f"# {title}\n\n"
    markdown_content += f"## Abstract\n\n{abstract}\n\n"
    if authors:
        markdown_content += f"## Authors\n\n" + ", ".join(authors) + "\n\n"

    for heading_level, section_title, section_content, section_id in sections:
        markdown_content += f"{'#' * heading_level} {section_title}\n\n{section_content}\n\n"

    # Save as Markdown
    with open(ieee_paper_info["output_md_path"], 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"Markdown successfully saved to {ieee_paper_info['output_md_path']}")

    ieee_paper_info['img_info'] = fig_table_data
    ieee_paper_info['section_info'] = sections
    return ieee_paper_info



def download_images(driver, ieee_paper_info):
    """
    셀레니움과 쿠키를 사용하여 이미지 다운로드 함수

    Args:
        img_dir (str): 이미지 다운로드 디렉토리 경로
        driver (webdriver): Selenium WebDriver 인스턴스
    """
    # 디렉토리 생성
    os.makedirs(ieee_paper_info['output_img_dir'], exist_ok=True)

    data = ieee_paper_info["img_info"]

    # Selenium에서 쿠키 가져오기
    cookies = driver.get_cookies()
    session = requests.Session()

    # requests 세션에 쿠키 추가
    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'])

    # 이미지 다운로드
    for item in data:
        image_url = item.get('image_href')
        file_name = item.get('img_file_name')

        if image_url and file_name:
            file_path = os.path.join(ieee_paper_info['output_img_dir'], file_name)
            try:
                response = session.get(image_url, stream=True)
                if response.status_code == 200:
                    with open(file_path, 'wb') as img_file:
                        for chunk in response.iter_content(1024):
                            img_file.write(chunk)
                    print(f"Downloaded: {file_path}")
                else:
                    print(f"Failed to download {image_url}, status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")




def extract_references(driver, ieee_paper_info):
    """
    HTML 파일에서 참조와 링크를 추출합니다.

    Args:
        file_path (str): 로컬 HTML 파일 경로

    Returns:
        list: 참조 번호, 제목, 링크 정보가 포함된 딕셔너리 리스트
    """

    driver.get(f"https://ieeexplore.ieee.org/document/{ieee_paper_info['ieee_paper_id']}/references#references")

    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

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

    ieee_paper_info['reference_info'] = reference_list
    return ieee_paper_info

def main():
    # Execute the function
    ieee_paper_info = {
        "output_md_path": 'test/output.md',
        "output_img_dir": "test/img",
        "relative_img_dir": "img",
        "paper_info_path": "test/paper_info.json",
        "ieee_paper_id": 8686550,
        "reference_info": [],
        "img_info": [],
        "section_info": [],
    }

    driver = webdriver.Chrome()  # 필요에 따라 WebDriver 경로 지정

    # step 1 : get reference data
    ieee_paper_info = extract_references(driver, ieee_paper_info)

    # step 2 : get markdown data
    ieee_paper_info = html_to_markdown(driver, ieee_paper_info)

    # step 3 : download images
    download_images(driver, ieee_paper_info)

    # step 4 :


    # step 4 : save markdown file

    # step 5 : save ieee_paper_info as json file
    with open(ieee_paper_info["paper_info_path"], 'w', encoding='utf-8') as json_file:
        json.dump(ieee_paper_info, json_file, ensure_ascii=False, indent=4)

main()
