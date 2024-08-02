import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset with line limit.')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of lines in the dataset.')
    args = parser.parse_args()

    if args.limit <= 0:
        args.limit = 40000000 + 34004228

    print("Downloading data...")
    os.makedirs("dataset", exist_ok=True)
    os.chdir("dataset")

    os.system("wget -nc https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json")
    os.system("wget -nc https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt")
    os.system("wget -nc https://storage.googleapis.com/huggingface-nlp/datasets/bookcorpus/bookcorpus.tar.bz2")

    print("Extracting data...")
    if not Path("bookcorpus.tar.bz2").exists():
        print("Error: bookcorpus.tar.bz2 not found.")
        exit(1)
    if not Path("books_large_p2.txt").exists():
        os.system("tar -xvjf bookcorpus.tar.bz2")
    else:
        print("Data already extracted.")

    print("Generating dataset...")
    with open("output.json", "w") as f2:
        line_count = 0
        for i in range(1, 3):
            print(f"Processing books_large_p{i}.txt")
            with open(f"books_large_p{i}.txt", "r") as f:
                for line in tqdm(f, total=(args.limit - line_count) if args.limit else None):
                    if args.limit and line_count >= args.limit:
                        break
                    text_line = line.rstrip()
                    if text_line:
                        f2.write(json.dumps({"text": text_line}))
                        f2.write("\n")
                    line_count += 1

    print("Processing dataset...")
    os.system(f"python ../preprocess_data.py --input output.json --tokenizer-model tokenizer.model --dataset-impl mmap --tokenizer-type GPTSentencePieceTokenizer --output-prefix=llama_training_data --append-eod --workers {os.cpu_count()}")
    os.chdir("..")