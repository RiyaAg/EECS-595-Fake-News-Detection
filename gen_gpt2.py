import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load LIAR dataset from Hugging Face
dataset = load_dataset("liar")

# Define model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained(model_name)

# Fine-tuning parameters
training_args = TrainingArguments(
    output_dir="./liar-finetuned-generation",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    label_smoothing_factor=1,
)

# Define data preprocessing function for text generation
def tokenize_function(examples):
    # for i in range(len(examples['statement'])):
    #     examples['statement'][i] = '<sos>' + examples["statement"][i] + '<eos>'
    return tokenizer(examples["statement"], return_tensors="pt", padding=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function)#, batched=True, batch_size=4)
tokenized_datasets = tokenized_datasets.map(lambda example: {'labels': example['input_ids']}, remove_columns=['label'])
# tokenized_datasets = tokenized_datasets.map(lambda example: {'input_ids': example['input_ids'][0][:10], 'attention_mask': example['attention_mask'][0][:10]})


# adjust dataset to only provide truncated input and let it learn to predict full string
# for i in ('test',):#('train', 'test', 'validation'):
#     # tokenized_datasets[i].pop('labels')
#     # tokenized_datasets[i].remove_columns('label')
#     for j in range(len(tokenized_datasets[i]['input_ids'])):
#         # tokenized_datasets[i]['label'][j] = tokenized_datasets[i]['input_ids'][j]
#         tokenized_datasets[i]['input_ids'][j][0] = tokenized_datasets[i]['input_ids'][j][0][:10]
    
# Define training function for text generation
def model_init():
    return GPT2LMHeadModel.from_pretrained(model_name)

# Instantiate Trainer for text generation
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
)

# Fine-tune the model for text generation
trainer.train()

# Save the fine-tuned text generation model
model.save_pretrained("./liar-finetuned-generation")
tokenizer.save_pretrained("./liar-finetuned-generation")
