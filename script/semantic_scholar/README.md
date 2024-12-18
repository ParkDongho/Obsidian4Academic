# Seamantic scholar integration

## 새로운 논문 정보 가져오기 

```bash
python3 get_new_paper_info.py
```

```
python3 get_new_paper_info.py --help

usage: get_new_paper_info.py [-h] [--output-dir OUTPUT_DIR] [--s2id-file S2ID_FILE] [--csv-file CSV_FILE]

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR, -d OUTPUT_DIR
                        Directory to save output files
  --s2id-file S2ID_FILE
                        Path to the S2ID file
  --csv-file CSV_FILE   Path to the journal list CSV file
```


## 논문 인용 데이터 가져오기 

```bash
python get_citation_info.py
python get_reference_info.py
```

```
usage: get_reference_info.py [-h] [--reference_info_dir REFERENCE_INFO_DIR] [--paper_dir PAPER_DIR] [--mode {missing,all}]

Find missing reference JSON files and fetch references.

options:
  -h, --help            show this help message and exit
  --reference_info_dir REFERENCE_INFO_DIR
                        Directory to write the reference JSON files.
  --paper_dir PAPER_DIR
                        Directory containing the paper files.
  --mode {missing,all}  Mode to fetch references: 'missing' fetches only missing ones, 'all' fetches all papers.
```









## 논문 리뷰 생성하기 










