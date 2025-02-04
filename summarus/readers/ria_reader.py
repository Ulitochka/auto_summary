import json
import csv
from typing import Dict

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter

from summarus.readers.summarization_reader import SummarizationReader


@DatasetReader.register("ria")
class RIAReader(SummarizationReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: int = 400,
                 target_max_tokens: int = 100,
                 separate_namespaces: bool = False,
                 target_namespace: str = "target_tokens",
                 save_copy_fields: bool = False,
                 save_pgn_fields: bool = False) -> None:

        if not tokenizer:
            tokenizer = WordTokenizer(word_splitter=SimpleWordSplitter())

        super().__init__(
            tokenizer=tokenizer,
            source_token_indexers=source_token_indexers,
            target_token_indexers=target_token_indexers,
            source_max_tokens=source_max_tokens,
            target_max_tokens=target_max_tokens,
            separate_namespaces=separate_namespaces,
            target_namespace=target_namespace,
            save_copy_fields=save_copy_fields,
            save_pgn_fields=save_pgn_fields
        )

    def parse_set(self, path):
        with open(path) as f:
            lines = csv.DictReader(f)
            next(lines, None)
            for el in lines:
                title = el['title']
                text = el['abstract']
                yield text, title
