import spacy
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

# Function to add dependency parsing features
def add_dependency_features(examples):
    texts = examples['text']
    dep_features = []
    for text in texts:
        doc = nlp(text)
        deps = [token.dep_ for token in doc]
        dep_features.append(" ".join(deps))
    examples['deps'] = dep_features
    return examples

# Load dataset
dataset = load_dataset('imdb')

# Add dependency parsing features to the dataset
dataset = dataset.map(add_dependency_features, batched=True)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save the model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')


# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./saved_model')
tokenizer = BertTokenizer.from_pretrained('./saved_model')

# Define a function to tokenize the test data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Tokenize the test dataset
tokenized_test_dataset = dataset['test'].map(tokenize_function, batched=True)

# Evaluate the model
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
)
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
