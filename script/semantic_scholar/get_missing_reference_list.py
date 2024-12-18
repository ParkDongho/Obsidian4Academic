import os

# Define the directory and file paths
json_dir = 'reference_json/'
paper_list_file = 'paper_list.txt'
output_file = 'missing_reference.txt'

# Get the list of JSON filenames without the '-citation' and '.json' extensions
json_files = [f.replace('-reference', '').replace('.json', '') for f in os.listdir(json_dir) if f.endswith('.json')]

# Read the paper list from the text file
with open(paper_list_file, 'r') as f:
    paper_list = [line.strip() for line in f.readlines()]

# Find papers in the list that don't have a corresponding JSON file
missing_papers = [paper for paper in paper_list if paper not in json_files]

# Write the missing papers to the output file
with open(output_file, 'w') as f:
    for paper in missing_papers:
        f.write(paper + '\n')

print(f"Missing papers have been written to {output_file}.")

