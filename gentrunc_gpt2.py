import torch
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer

dataset = load_dataset('newsroom')

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", force_bos_token_to_be_generated=True)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

