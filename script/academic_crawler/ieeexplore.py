from bs4 import BeautifulSoup

# Load the HTML file
input_html_path = 'paper.html'
output_md_path = 'output.md'

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
    sections = []
    for section_class in ['section', 'section_2']:
        for section in soup.find_all('div', class_=section_class):
            # Handle headings from h2 to h6
            heading = None
            for level in range(2, 7):
                heading_tag = section.find(f'h{level}')
                if heading_tag:
                    heading = (level, heading_tag.text.strip())
                    break

            section_title = heading[1] if heading else "Untitled Section"
            section_content = ""

            # Process paragraphs and nested content
            for paragraph in section.find_all('p'):
                # Replace inline and display formulas with Markdown-friendly syntax
                for math in paragraph.find_all('inline-formula'):
                    latex_code = math.find('tex-math').text if math.find('tex-math') else ""
                    math.replace_with(f"${latex_code}$")

                for math in paragraph.find_all('display-formula'):
                    latex_code = math.find('tex-math').text if math.find('tex-math') else ""
                    math.replace_with(f"\n$$\n{latex_code}\n$$\n")

                # Add paragraph content with spaces between elements
                paragraph_text = ' '.join(paragraph.stripped_strings)
                section_content += paragraph_text + "\n\n"

            # Process images
            for img in section.find_all('img'):
                img_src = img.get('src', '')
                img_alt = img.get('alt', 'Image')
                section_content += f"![{img_alt}]({img_src})\n\n"

            # Process nested subsections
            for subsection in section.find_all('div', recursive=False):
                subsection_heading = None
                for level in range(3, 7):
                    heading_tag = subsection.find(f'h{level}')
                    if heading_tag:
                        subsection_heading = (level, heading_tag.text.strip())
                        break

                if subsection_heading:
                    subsection_content = ""
                    for paragraph in subsection.find_all('p'):
                        paragraph_text = ' '.join(paragraph.stripped_strings)
                        subsection_content += paragraph_text + "\n\n"

                    for img in subsection.find_all('img'):
                        img_src = img.get('src', '')
                        img_alt = img.get('alt', 'Image')
                        subsection_content += f"![{img_alt}]({img_src})\n\n"

                    sections.append((subsection_heading[0], subsection_heading[1], subsection_content))

            sections.append((heading[0] if heading else 2, section_title, section_content))

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

