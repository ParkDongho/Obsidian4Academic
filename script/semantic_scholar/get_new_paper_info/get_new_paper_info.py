#!/usr/bin/env python3
import dotenv

dotenv.load_dotenv()
import argparse
import os
from requests import Session
import time
import yaml

from script.semantic_scholar.get_new_paper_info import load_journal_dict, create_yaml, get_paper_metadata

S2_API_KEY = os.environ.get('S2_API_KEY', '')

def get_new_paper_info(args: argparse.Namespace, journal_dict) -> None:
    with open(args.s2id_file, 'r') as s2id_file:
        s2ids = [line.strip() for line in s2id_file.readlines()]
    fields = 'title,authors,year,venue,abstract,citationCount,externalIds,publicationDate'
    for paper_id in s2ids:
        with Session() as session:
            paper_metadata = get_paper_metadata(session, paper_id, fields=fields)

        if not paper_metadata:
            print(f'No metadata found for paper ID {paper_id}')
            continue

        yaml_content = create_yaml(paper_metadata, journal_dict, paper_id)
        output_filename = f'{paper_id}.yaml'
        with open(output_filename, 'w') as yamlfile:
            yaml.dump(yaml_content, yamlfile, default_flow_style=False, allow_unicode=True)

        time.sleep(3)
        print(f'Wrote YAML for paper ID {paper_id} to {output_filename}')


def get_and_save_paper_info():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output-dir',
        default = '/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/.paper_info/',
        help="Directory to save output files"
    )
    parser.add_argument(
        '--s2id-file',
        default='/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/new_paper_list.txt',
        help="Path to the S2ID file"
    )
    parser.add_argument(
        '--csv-file',
        default='/home/parkdongho/dev/Obsidian4Academic/20_Works/21_Research/1_paper_archive/journal_list.csv',
        help="Path to the journal list CSV file"
    )

    args = parser.parse_args()

    # Load the journal dictionary from the CSV file
    journal_dict = load_journal_dict(args.csv_file)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Change working directory to output directory
    os.chdir(args.output_dir)

    get_new_paper_info(args, journal_dict)


