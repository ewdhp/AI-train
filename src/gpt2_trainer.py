import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_linear_schedule_with_warmup
import logging

class GPT2Trainer:
    def __init__(self, model_name='gpt2', use_validation=False, device='auto'):
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("GPT2Trainer")

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Select device
        self.device = self._select_device(device)
        self.model.to(self.device)

        self.use_validation = use_validation

    def _select_device(self, device):
        """Select the appropriate device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                self.logger.info("Using NVIDIA GPU")
                return torch.device('cuda')
            try:
                import openvino.runtime as ov
                self.logger.info("Using Intel OpenVINO")
                return ov.Core()  # Placeholder: OpenVINO usage would require conversion of model.
            except ImportError:
                self.logger.warning("Intel OpenVINO not available. Falling back to CPU.")
                return torch.device('cpu')
        else:
            self.logger.info(f"Using specified device: {device}")
            return torch.device(device)

    def save_checkpoint(self, epoch, step, running_loss, val_loss, checkpoint_dir='checkpoints'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'running_loss': running_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        running_loss = checkpoint['running_loss']
        val_loss = checkpoint['val_loss']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return epoch, step, running_loss, val_loss

    def train_manual_grad(self, dataset, epochs=1, batch_size=1, learning_rate=5e-5, 
                          accumulation_steps=4, validation_split='validation', 
                          checkpoint_dir='checkpoints', resume_from_checkpoint=None, 
                          use_best_checkpoint=False):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = None

        self.model.train()

        # Scheduler with warm-up
        total_steps = len(dataset['train']) * epochs // batch_size
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=total_steps // 10, 
                                                         num_training_steps=total_steps)

        # Validation split handling
        if self.use_validation and validation_split not in dataset:
            train_size = int(0.8 * len(dataset['train']))
            val_size = len(dataset['train']) - train_size
            train_dataset, val_dataset = random_split(dataset['train'], [train_size, val_size])
            dataset = {'train': train_dataset, 'validation': val_dataset}

        dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset[validation_split], batch_size=batch_size, shuffle=False) if self.use_validation else None

        start_epoch = 0
        start_step = 0
        running_loss = 0.0
        best_val_loss = float('inf')

        if resume_from_checkpoint:
            checkpoint_path = self.get_best_checkpoint(checkpoint_dir) if use_best_checkpoint else self.get_last_checkpoint(checkpoint_dir)
            if checkpoint_path:
                start_epoch, start_step, running_loss, best_val_loss = self.load_checkpoint(checkpoint_path)

        self.logger.info(f"Total training samples: {len(dataset['train'])}")
        self.logger.info(f"Steps per epoch: {len(dataloader)}")

        for epoch in range(start_epoch, epochs):
            for i, batch in enumerate(dataloader):
                if epoch == start_epoch and i < start_step:
                    continue

                concatenated = [p + " " + r for p, r in zip(batch['prompt'], batch['response'])]
                inputs = self.tokenizer(concatenated, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(self.device)
                labels = inputs.input_ids.clone().to(self.device)

                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps
                loss.backward()

                running_loss += loss.item()

                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(f"Epoch {epoch + 1}, Step {i + 1}, Loss: {running_loss:.4f}, LR: {current_lr:.6f}")
                    running_loss = 0.0

                    # Save checkpoint
                    if self.use_validation:
                        val_loss = self.evaluate(val_dataloader)
                        self.save_checkpoint(epoch, i + 1, running_loss, val_loss, checkpoint_dir)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_checkpoint(epoch, i + 1, running_loss, val_loss, checkpoint_dir)

            if self.use_validation:
                val_loss = self.evaluate(val_dataloader)
                self.logger.info(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        self.model.save_pretrained('gpt2_model')
        self.tokenizer.save_pretrained('gpt2_tokenizer')

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in dataloader:
                concatenated = [p + " " + r for p, r in zip(batch['prompt'], batch['response'])]
                inputs = self.tokenizer(concatenated, return_tensors='pt', padding=True, truncation=True)
                inputs = inputs.to(self.device)
                labels = inputs.input_ids.clone().to(self.device)

                outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
                total_loss += outputs.loss.item()

        avg_loss = total_loss / len(dataloader)
        self.model.train()
        return avg_loss
