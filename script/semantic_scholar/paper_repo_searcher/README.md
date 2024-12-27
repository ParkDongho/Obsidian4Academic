# 기능 

ieee paper 번호를 입력하면 해당 논문의 semantic scholar id를 반환
- `.yaml` 파일을 읽어서 ieee paper 번호에서 semantic scholar id를 얻어냄
- 만약 해당 논문에 대한 `.yaml` 파일이 없다면,
  - ieee paper 번호를 통해 doi를 얻어내고 (`get_doi_from_ieee_paper`)
  - doi를 통해 semantic scholar id를 얻어냄 (`get_semantic_scholar_id_from_doi`)
  - semantic scholar 에 대한 `paper_id.yaml` 파일을 생성


# 함수 목록 

## get_doi_from_ieee_id

ieee id를 입력하면 해당 논문의 doi를 반환

```python
from selenium import webdriver
from script.semantic_scholar.paper_repo_searcher.get_doi_from_ieee_id import get_doi_from_ieee_id

driver = webdriver.Chrome()
doi = get_doi_from_ieee_id(123456, driver)
```

```bash
# 7738524 -> https://doi.org/10.1109/JSSC.2016.2616357
python3 get_doi_from_ieee_id.py 
```

## get_semantic_scholar_id_from_doi

```bash
# 10.1109/JSSC.2016.2616357 -> https://www.semanticscholar.org/paper/ffdaa12ef011de9dbf43be46d45a3abcc8288965
python3 get_semantic_id_from_doi.py 
```

## get_semantic_id_from_ieee_id

```bash
python3 get_semantic_id_from_ieee_id.py 
```

