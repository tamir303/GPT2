import os
from typing import List, Union

import sentencepiece as spm
import torch


class Tokenizer:
    __options = dict(
        input_format="text",
        model_prefix="tok400",  # output filename prefix
        model_type="bpe",
        vocab_size=1000,
        normalization_rule_name="identity",  # ew, turn off normalization
        remove_extra_whitespaces=False,
        input_sentence_size=200000000,  # max number of training sentences
        max_sentence_length=4192,  # max number of bytes per sentence
        seed_sentencepiece_size=1000000,
        shuffle_input_sentence=True,
        character_coverage=0.99995,
        byte_fallback=True,
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        max_sentencepiece_length=16,
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=-1,
        num_threads=os.cpu_count(),  # use ~all system resources
    )

    __model_file = 'tok400.model'

    def __init__(self, file_path: str):
        self.sp = spm.SentencePieceProcessor()
        try:
            self.sp.Load(self.__model_file)
        except Exception as e:
            self.__options["input"] = file_path
            spm.SentencePieceTrainer.Train(**self.__options)
        finally:
            self.sp.Load(self.__model_file)

    def encode(self, raw: Union[str, List[str]]) -> torch.Tensor:
        return torch.tensor([self.sp.Encode(raw)])

    def decode(self, token_ids: torch.Tensor) -> Union[str, List[str]]:
        return self.sp.Decode(token_ids)

    def get_vocab_size(self) -> int:
        return self.sp.vocab_size()
