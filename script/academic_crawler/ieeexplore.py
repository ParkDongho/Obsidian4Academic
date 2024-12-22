from bs4 import BeautifulSoup

# Load the HTML file
input_html_path = 'paper.html'
output_md_path = 'output.md'

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

            # 본문 처리
            elif paragraph.name == "p":
                # Replace inline and display formulas with Markdown-friendly syntax
                for math in paragraph.find_all('inline-formula'):
                    latex_code = math.find('tex-math').text if math.find('tex-math') else ""
                    math.replace_with(f"${latex_code}$")

                # link 처리

                paragraph_text = ' '.join(paragraph.stripped_strings)
                section_content += paragraph_text + "\n\n"

            elif paragraph.name == "disp-formula":
                # 찾고자 하는 <span> 태그와 그 내부의 텍스트를 추출
                span_tag = paragraph.find('span', class_="tex tex2jax_ignore")
                if span_tag and span_tag.text:
                    latex_code = span_tag.text.strip()  # 텍스트를 가져와서 공백 제거
                    section_content += f"$$\n{latex_code}\n$$\n\n"

            # Figure 처리
            elif any(cls in paragraph.get('class', []) for cls in ["figure", "figure-full"]):
                print("figure")

            # Table 처리


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

