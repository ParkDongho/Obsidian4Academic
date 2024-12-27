import os
import pandas as pd

S2_API_KEY = os.environ.get('S2_API_KEY', '')

def load_journal_dict(csv_file_path):
    # CSV 파일을 읽어 'journal'과 'short' 컬럼을 딕셔너리로 반환
    df = pd.read_csv(csv_file_path)
    return dict(zip(df['journal'], df['name_short']))


