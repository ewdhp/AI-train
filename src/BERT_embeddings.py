from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2).item()

# Initialize BERT tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Sentences to compare
sentence_1 = "BERT embeddings capture semantic meaning."
sentence_2 = "BERT represents the meaning of words in context."
sentence_3 = "Apples and oranges are fruits."

# Tokenize and prepare inputs
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    """
    Instead of just using the [CLS] token, average all token 
    embeddings to represent the sentence. This can sometimes 
    provide a more balanced representation:
    """
    sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1)

    return sentence_embedding

# Get embeddings
embedding_1 = get_embedding(sentence_1)
embedding_2 = get_embedding(sentence_2)
embedding_3 = get_embedding(sentence_3)

# Compute similarities
similarity_1_2 = cosine_similarity(embedding_1, embedding_2)
similarity_1_3 = cosine_similarity(embedding_1, embedding_3)

# Display results
print(f"Similarity between Sentence 1 and Sentence 2: {similarity_1_2:.4f}")
print(f"Similarity between Sentence 1 and Sentence 3: {similarity_1_3:.4f}")
