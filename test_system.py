import torch
from src.model import EmotionModel
from src.config import Config

def test_system():
    print("Testing model forward pass...")
    model = EmotionModel("microsoft/wavlm-base") # use base for speed in test
    input_values = torch.randn(2, 16000 * 2) # 2 seconds, batch 2
    
    logits, vad = model(input_values)
    
    print("Logits shape:", logits.shape)
    print("VAD shape:", vad.shape)
    
    assert logits.shape == (2, 4)
    assert vad.shape == (2, 3)
    print("Test passed!")

if __name__ == "__main__":
    test_system()
