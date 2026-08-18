import os
from datasets import Dataset

def load_text_files_from_folder(folder_path="./data"):
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append({"text": content})
    return Dataset.from_list(texts)

if __name__ == "__main__":
    dataset = load_text_files_from_folder()
    print(f"Loaded {len(dataset)} files from data folder.")
