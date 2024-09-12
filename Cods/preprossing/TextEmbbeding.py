
from typing import Optional

from transformers import AutoTokenizer, AutoModel
import os
import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizerBase
import json



class preprossesor():
    def __int__(self ,text):
        self.text = text

    def fix_persian_kaf_yeh(self):
        """
        Generate a normalizer function that removes and replace many non-standard characters!
        """
        persian_yeh = 'ی'  # ARABIC LETTER FARSI YEH      U+06CC
        persian_kaf = 'ک'  # ARABIC LETTER KEHEH          U+06A9

        normalize_replace_map = {
            'ي': persian_yeh,  # ARABIC LETTER YEH                   U+064A
            'ى': persian_yeh,  # ARABIC LETTER ALEF MAKSURA          U+0649
            'ك': persian_kaf,  # ARABIC LETTER KAF                   U+0643
            'ڪ': persian_kaf,  # Arabic Letter Swash Kaf             U+06AA

            'ـ': None  # ARABIC TATWEEL                      U+0640
        }

        trans_dict = str.maketrans(normalize_replace_map)
        return self.text.translate(trans_dict)

class ModelUtils :
    def __init__(self, model_root) :
        self.model_root = model_root
        self.model_path = os.path.join(model_root, "model")
        self.tokenizer_path = os.path.join(model_root, "tokenizer")

    def download_model (self) :
        BASE_MODEL = "HooshvareLab/bert-fa-zwnj-base"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModel.from_pretrained(BASE_MODEL)

        tokenizer.save_pretrained(self.tokenizer_path)
        model.save_pretrained(self.model_path)

    def make_dirs (self) :
        if not os.path.isdir(self.model_root) :
            os.mkdir(self.model_root)
        if not os.path.isdir(self.model_path) :
            os.mkdir(self.model_path)
        if not os.path.isdir(self.tokenizer_path) :
            os.mkdir(self.tokenizer_path)

class Preprocess :
    def __init__(self, model_root) :
        self.model_root = model_root
        self.model_path = os.path.join(model_root, "model")
        self.tokenizer_path = os.path.join(model_root, "tokenizer")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def vectorize (self, text) :
        model = AutoModel.from_pretrained(self.model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        ids, masks = self.transform_single_text(text, tokenizer, 510, stride=510, minimal_chunk_length=0, maximal_text_length=None)
        # ids = torch.cat(ids, dim=0)
        # masks = torch.cat(masks, dim=0)
        tokens = {'input_ids': ids.to(self.device), 'attention_mask': masks.to(self.device)}

        output = model(**tokens)
        last_hidden_states = output.last_hidden_state

        # first token embedding of shape <1, hidden_size>
        # first_token_embedding = last_hidden_states[:,0,:]

        # pooled embedding of shape <1, hidden_size>
        mean_pooled_embedding = last_hidden_states.mean(axis=1)

        result = mean_pooled_embedding.flatten().cpu().detach().numpy()
        # print(result.shape)
        # print(result)
        # Convert the list to JSON
        json_data = json.dumps(result.tolist())

        return json_data



    def transform_list_of_texts(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        maximal_text_length: Optional[int] = None,
    ) -> BatchEncoding:
        model_inputs = [
            self.transform_single_text(text, tokenizer, chunk_size, stride, minimal_chunk_length, maximal_text_length)
            for text in texts
        ]
        input_ids = [model_input[0] for model_input in model_inputs]
        attention_mask = [model_input[1] for model_input in model_inputs]
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        return input_ids, attention_mask


    def transform_single_text(
        self,
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
        maximal_text_length: Optional[int],
    ) -> tuple[Tensor, Tensor]:
        """Transforms (the entire) text to model input of BERT model."""
        if maximal_text_length:
            tokens = self.tokenize_text_with_truncation(text, tokenizer, maximal_text_length)
        else:
            tokens = self.tokenize_whole_text(text, tokenizer)
        input_id_chunks, mask_chunks = self.split_tokens_into_smaller_chunks(tokens, chunk_size, stride, minimal_chunk_length)
        self.add_special_tokens_at_beginning_and_end(input_id_chunks, mask_chunks)
        self.add_padding_tokens(input_id_chunks, mask_chunks)
        input_ids, attention_mask = self.stack_tokens_from_all_chunks(input_id_chunks, mask_chunks)
        return input_ids, attention_mask


    def tokenize_whole_text(self, text: str, tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
        """Tokenizes the entire text without truncation and without special tokens."""
        tokens = tokenizer(text, add_special_tokens=False, truncation=False, return_tensors="pt")
        return tokens


    def tokenize_text_with_truncation(
        self, text: str, tokenizer: PreTrainedTokenizerBase, maximal_text_length: int
    ) -> BatchEncoding:
        """Tokenizes the text with truncation to maximal_text_length and without special tokens."""
        tokens = tokenizer(
            text, add_special_tokens=False, max_length=maximal_text_length, truncation=True, return_tensors="pt"
        )
        return tokens


    def split_tokens_into_smaller_chunks(
        self,
        tokens: BatchEncoding,
        chunk_size: int,
        stride: int,
        minimal_chunk_length: int,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Splits tokens into overlapping chunks with given size and stride."""
        input_id_chunks = self.split_overlapping(tokens["input_ids"][0], chunk_size, stride, minimal_chunk_length)
        mask_chunks = self.split_overlapping(tokens["attention_mask"][0], chunk_size, stride, minimal_chunk_length)
        return input_id_chunks, mask_chunks


    def add_special_tokens_at_beginning_and_end(self, input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
        """
        Adds special CLS token (token id = 101) at the beginning.
        Adds SEP token (token id = 102) at the end of each chunk.
        Adds corresponding attention masks equal to 1 (attention mask is boolean).
        """
        for i in range(len(input_id_chunks)):
            # adding CLS (token id 101) and SEP (token id 102) tokens
            input_id_chunks[i] = torch.cat([Tensor([101]), input_id_chunks[i], Tensor([102])])
            # adding attention masks  corresponding to special tokens
            mask_chunks[i] = torch.cat([Tensor([1]), mask_chunks[i], Tensor([1])])


    def add_padding_tokens(self, input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> None:
        """Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens."""
        for i in range(len(input_id_chunks)):
            # get required padding length
            pad_len = 512 - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([input_id_chunks[i], Tensor([0] * pad_len)])
                mask_chunks[i] = torch.cat([mask_chunks[i], Tensor([0] * pad_len)])


    def stack_tokens_from_all_chunks(self, input_id_chunks: list[Tensor], mask_chunks: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Reshapes data to a form compatible with BERT model input."""
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)

        return input_ids.long(), attention_mask.int()


    def split_overlapping(self, tensor: Tensor, chunk_size: int, stride: int, minimal_chunk_length: int) -> list[Tensor]:
        """Helper function for dividing 1-dimensional tensors into overlapping chunks."""
        self.check_split_parameters_consistency(chunk_size, stride, minimal_chunk_length)
        result = [tensor[i : i + chunk_size] for i in range(0, len(tensor), stride)]
        if len(result) > 1:
            # ignore chunks with less than minimal_length number of tokens
            result = [x for x in result if len(x) >= minimal_chunk_length]
        return result


    def check_split_parameters_consistency(self, chunk_size: int, stride: int, minimal_chunk_length: int) -> None:
        if chunk_size > 510:
            raise RuntimeError("Size of each chunk cannot be bigger than 510!")
        if minimal_chunk_length > chunk_size:
            raise RuntimeError("Minimal length cannot be bigger than size!")
        if stride > chunk_size:
            raise RuntimeError(
                "Stride cannot be bigger than size! Chunks must overlap or be near each other!"
            )
