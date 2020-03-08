import os
import csv
import json
import tempfile
import argparse

from sentencepiece import SentencePieceTrainer as sp_trainer

TEXT_LENS = []
TITLE_LENS = []


def read(path):
    """Чтение файла csv"""
    with open(path) as f:
        lines = csv.DictReader(f)
        next(lines, None)
        for el in lines:
            title = el['title']
            text = el['abstract']

            tokens_t = len(text.split())
            tokens_tt = len(title.split())

            TEXT_LENS.append(tokens_t)
            TITLE_LENS.append(tokens_tt)

            yield text, title
   

def train_subwords(train_path, model_path, model_type, vocab_size):
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False)
    for text, title in read(train_path):
        temp.write(text + "\n")
        temp.write(title + "\n")
    temp.close()
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    cmd = "--input={} --model_prefix={} --vocab_size={} --model_type={}".format(
        temp.name,
        os.path.join(model_path, model_type),
        vocab_size,
        model_type)
    sp_trainer.Train(cmd)
    os.unlink(temp.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-path', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default="bpe")
    parser.add_argument('--vocab-size', type=int, default=50000)
    args = parser.parse_args()
    train_subwords(**vars(args))

    print(min(TEXT_LENS), max(TEXT_LENS), sum(TEXT_LENS) / len(TEXT_LENS))
    print(min(TITLE_LENS), max(TITLE_LENS), sum(TITLE_LENS) / len(TITLE_LENS))
