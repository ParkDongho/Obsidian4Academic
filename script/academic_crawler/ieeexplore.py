from bs4 import BeautifulSoup

# Load the HTML file
input_html_path = 'paper.html'
output_md_path = 'output.md'
ieee_paper_id = 1

from bs4 import Tag, NavigableString

def parsePaper(input):
    sections = []
    for section in input:
        single_section = parseSection(section)
        sections = sections + single_section
    return sections


def parseSection(input):

    single_section = []
    subsec_list = []
    heading_level = None
    section_title = ""
    section_content = ""

    for paragraph in input.contents:
        # Tag인지 NavigableString인지 확인
        if isinstance(paragraph, NavigableString):
            continue

        # subsection 처리
        if 'section_2' in paragraph.get('class', "div"):
            subsec_list = subsec_list + parseSection(paragraph)

        else:
            if (paragraph.name in ["h3", "h4", "h5", "h6"]) or ('header' in paragraph.get('class', "div")):
                for level in range(2, 7):
                    heading_tag = input.find(f'h{level}')
                    if heading_tag:
                        heading_level = level
                        section_title = heading_tag.text.strip()
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
            elif any(cls in paragraph.get('class', []) for cls in ["figure", "figure-full"]):
                # 이미지와 캡션 파싱
                image_wrap = paragraph.find('div', class_='img-wrap')
                fig_caption = paragraph.find('div', class_='figcaption')

                if image_wrap and fig_caption:
                    # 링크와 이미지 경로 추출
                    link_tag = image_wrap.find('a')
                    img_tag = link_tag.find('img') if link_tag else None
                    image_href = link_tag['href'] if link_tag else ''
                    alt_text = img_tag.get('alt', 'Image') if img_tag else ''
                    data_fig_id = link_tag.get('data-fig-id') if img_tag else ''

                    # 캡션 추출
                    caption_title = fig_caption.find('b', class_='title').get_text(strip=True) if fig_caption.find('b',
                                                                                                                   class_='title') else ''
                    caption_text = fig_caption.find('p').get_text(strip=True) if fig_caption.find('p') else ''

                    # 마크다운 형식으로 변환
                    markdown_output = f"![{alt_text}](ieee_{ieee_paper_id}_{data_fig_id}.gif)\n\n**{caption_title}** {caption_text}"

                    # 섹션 내용에 추가
                    section_content += f"\n{markdown_output}\n"

                else:
                    section_content += "\nFigure could not be parsed.\n"


            # Table 생성
            elif any(cls in paragraph.get('class', []) for cls in ["table", "table-full"]):
                print("table")


            else:
                print("\n\nUnhandled Tag:", paragraph)


        single_section = [(heading_level, section_title, section_content)] + subsec_list

    return single_section

# Function to extract content and convert to Markdown
def html_to_markdown(html_path, markdown_path):
    with open(html_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract title
    title = soup.find('title').text.strip() if soup.find('title') else "No Title"

    # Extract abstract
    abstract_section = soup.find('meta', {'name': 'Description'})
    abstract = abstract_section['content'] if abstract_section else "No Abstract"

    # Extract authors
    authors = [tag['content'] for tag in soup.find_all('meta', {'name': 'parsely-author'})]

    # Extract sections and process paragraphs
    tmp = soup.find_all('div', class_=['section'])
    sections = parsePaper(tmp)

    # Convert to Markdown
    markdown_content = f"# {title}\n\n"
    markdown_content += f"## Abstract\n\n{abstract}\n\n"
    if authors:
        markdown_content += f"## Authors\n\n" + ", ".join(authors) + "\n\n"

    for heading_level, section_title, section_content in sections:
        markdown_content += f"{'#' * heading_level} {section_title}\n\n{section_content}\n\n"

    # Save as Markdown
    with open(markdown_path, 'w', encoding='utf-8') as md_file:
        md_file.write(markdown_content)

    print(f"Markdown successfully saved to {markdown_path}")

# Execute the function
html_to_markdown(input_html_path, output_md_path)

