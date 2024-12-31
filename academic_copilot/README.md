# Academic Copilot

## Overview 



## Installation



## Project Structure

- academic_crawler
  - [x] **ieeexplore.py** : 
  - [ ] acm.py : <!-- TODO -->
  - [ ] arxiv.py : <!-- TODO -->
- document_generator
  - [ ] pdf_to_text.py : <!-- TODO -->
  - [ ] text_to_pdf.py : <!-- TODO -->
  - [ ] text_to_ppt.py : <!-- TODO -->
  - [ ] text_to_slide.py : <!-- TODO -->
- gpt_integration
  - [ ] **ocr.py** : <!-- TODO -->
  - [ ] **text_gen.py** : <!-- TODO --> 
  - [ ] **slide_gen.py** : <!-- TODO --> 
  - [ ] **summarize.py** : <!-- TODO --> 
  - [x] **translate.py** : <!-- TODO --> 
- semantic_scholar : `academic_copilot/semantic_scholar`
  - **get_paper_info.py** semantic scholar api를 이용하여 논문 정보를 가져옴
    - [x] `save_paper_info_from_semantic_id(semantic_id)` : 
    - [x] `save_paper_info_from_paper_list(new_paper_list)` :
  - **get_biblio_info.py :** semantic scholar api를 이용하여 reference/citation 정보 및 인용 스타일을 추출 
    - [ ] `#TODO`: `get_citation_info.py` 와 `get_reference_info.py`를 하나의 파일(`get_biblio_info.py`) 로 결합 
    - [x] `get_citation_info.py` :  
    - [x] `get_reference_info.py` :   
  - **get_academic_id.py**
    - [x] `get_semantic_id_from_doi()` :   
    - [x] `get_semantic_id_from_ieee_id()` :   
    - [x] `get_journal_id_from_doi()` :   
    - [x] `get_doi_from_ieee_id(ieee_id, driver) -> doi_id` :  
  - **academic_database.py**
    - [x] `search_from_database(key, value, result_key) -> result_value` :  


