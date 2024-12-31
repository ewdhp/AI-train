import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from openvino.tools.mo import mo
from openvino.runtime import Core, Tensor

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, batch_first=True):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.positional_encoding = nn.Embedding(max_seq_length, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        batch_size, src_seq_len, _ = src.size()
        batch_size, tgt_seq_len, _ = tgt.size()
        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0).repeat(batch_size, 1)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).repeat(batch_size, 1)
        src = src + self.positional_encoding(src_pos)
        tgt = tgt + self.positional_encoding(tgt_pos)
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        output = self.linear(output)
        return output

# Hyperparameters
d_model = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
max_seq_length = 50
learning_rate = 0.001
epochs = 1

# Initialize the model
transformer_model = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)

# Define the optimizer
optimizer = optim.Adam(transformer_model.parameters(), lr=learning_rate)

# Dummy dataloader for demonstration purposes
train_dataloader = [(torch.randn(10, max_seq_length, d_model), torch.randn(10, max_seq_length, d_model)) for _ in range(10)]

# Loss function
criterion = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    print(f'Starting epoch {epoch+1}/{epochs}')
    for batch_idx, (src, tgt) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = transformer_model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item()}')
    print(f'Epoch {epoch+1}/{epochs} completed with Loss: {loss.item()}')


# Save the model
torch.save(transformer_model.state_dict(), 'transformer_model.pth')

# Load the model
loaded_model = Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length)
loaded_model.load_state_dict(torch.load('transformer_model.pth'))
loaded_model.eval()
print(output)
