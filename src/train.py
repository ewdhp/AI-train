from datasets import load_dataset
import torch
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, EarlyStoppingCallback
import logging


# Load the pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./gpt2_finetuned')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_finetuned')

input_prompt="The human-induced carbon dioxide by"

logging.info("Encoding input prompt...")
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
attention_mask = torch.ones(input_ids.shape, device=model.device)  # Create attention mask of ones

# Ensure the model is in evaluation mode for inference
model.eval()

# Move the input tensors to the same device as the model (CPU/GPU)
input_ids = input_ids.to(model.device)
attention_mask = attention_mask.to(model.device)

# Generate a response
logging.info("Generating response...")
# Generate a response with additional parameters
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,  # Maximum length for the output
    min_length=15,   # Ensure the generated text is at least 50 tokens long
    num_return_sequences=1,
    repetition_penalty=2.0,  # Penalize repeated tokens
)

# Decode and print the generated response
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
