import openai
import re
import tiktoken
import time
import os
import argparse
import json

# Memory file for storing previous questions and answers
MEMORY_FILE = "memory.json"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

def translate(text, memory, client):
    if text in memory:
        print("Using cached translation.")
        return memory[text]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Translate English text into Korean."},
            {"role": "user", "content": text}
        ]
    )
    translated_text = response.choices[0].message.content
    memory[text] = translated_text
    save_memory(memory)
    return translated_text

def read_and_split_texts(file_path, file_name):
    with open(file_path + file_name + ".txt", 'r', encoding='utf-8') as file:
        content = file.read()
        chapters = re.split(r'(# \d+\n|## \d+\n)', content)
        chapters = [chapters[i] + chapters[i+1] for i in range(1, len(chapters), 2)]
    return chapters

def save_text(file_path, file_name, file_content):
    os.makedirs(file_path, exist_ok=True)
    with open(file_path + file_name + ".txt", 'w', encoding='utf-8') as file:
        file.write(file_content)
    print(f"Saved: {file_path}{file_name}.txt")

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def split_text_by_length_into_n_parts(text, n):
    total_length = len(text)
    part_length = total_length // n
    parts = []
    start = 0

    for i in range(n):
        if i == n - 1:
            parts.append(text[start:])
            break

        end = start + part_length
        split_index = text.rfind("\n\n", start, end)
        if split_index == -1:
            split_index = end

        parts.append(text[start:split_index])
        start = split_index + 2

    return parts


def main():
    client = openai.OpenAI()
    parser = argparse.ArgumentParser(description="Translate and process text files.")
    parser.add_argument("--file_name", required=True, help="Name of the input text file (without extension)")
    parser.add_argument("--read_file_path", required=True, help="Path to the input file directory")
    parser.add_argument("--write_file_path", required=True, help="Path to the output file directory")
    parser.add_argument("--start_index", type=int, default=1, help="Starting chapter index")
    parser.add_argument("--end_index", type=int, default=100, help="Ending chapter index")
    parser.add_argument("--re_translate_threshold", type=int, default=100, help="Minimum length of translated text to trigger re-translation")

    args = parser.parse_args()

    memory = load_memory()

    chapter_texts = read_and_split_texts(args.read_file_path, args.file_name)
    print(f"Total chapters: {len(chapter_texts)}")

    for i, chapter in enumerate(chapter_texts[args.start_index - 1:args.end_index]):
        print("\n----------------------")
        print(f"Processing chapter {i + args.start_index}")
        token_num = count_tokens(chapter)
        print(f"Tokens: {token_num}")

        if token_num > 2000:
            n = token_num // 2000 + 1
            splited_chapter = split_text_by_length_into_n_parts(chapter, n)
            translated_text_list = []

            for j, part in enumerate(splited_chapter):
                print(f"Part {j + 1} tokens: {count_tokens(part)}")
                translated_text = translate(part, memory, client)
                translated_text_list.append(translated_text)

            translated_text = "\n\n".join(translated_text_list)
        else:
            translated_text = translate(chapter, memory, client)
            print(f"Translated text length: {len(translated_text)}")
            while len(translated_text) < args.re_translate_threshold:
                print("Re-translating...")
                time.sleep(5)
                translated_text = translate(chapter, memory, client)

        output_path = f"{args.write_file_path}/{args.file_name}/"
        save_text(output_path, f"{args.file_name}_{str(i + args.start_index).zfill(4)}", translated_text)
        print("----------------------\n")

if __name__ == "__main__":
    main()


