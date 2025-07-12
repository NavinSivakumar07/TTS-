import torch
import torch.nn as nn
from torch.optim import Adam
from src.data_loader import get_dataloader
from src.model import SimpleTTS
from src import config

def train():
    """
    Main training loop for the SimpleTTS model.
    """
    # Setup
    device = torch.device(config.DEVICE)
    model = SimpleTTS().to(device)
    dataloader = get_dataloader(config.DATA_PATH, config.BATCH_SIZE)
    
    # Loss and optimizer
    criterion = nn.L1Loss() # Mean Absolute Error is common for spectrograms
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print("Starting training...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Move data to the appropriate device
            text_padded, mel_target = [d.to(device) for d in batch]
            
            # Forward pass
            mel_pred = model(text_padded)
            
            # Calculate loss
            loss = criterion(mel_pred, mel_target)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_THRESH)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}], Loss: {avg_loss:.4f}")

    print("\\nTraining complete.")
    
    # Save the trained model
    torch.save(model.state_dict(), "simple_tts_model.pth")
    print("Model saved to simple_tts_model.pth")

if __name__ == '__main__':
    train()
