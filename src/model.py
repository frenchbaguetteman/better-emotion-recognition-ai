import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls = nn.Linear(in_dim, 1)
        
    def forward(self, x, attention_mask=None):
        # x: (B, T, D)
        scores = self.cls(x) # (B, T, 1)
        
        if attention_mask is not None:
            # attention_mask: (B, T)
            # Make sure mask is broadcastable
            mask = attention_mask.unsqueeze(-1) # (B, T, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        alpha = torch.softmax(scores, dim=1) # (B, T, 1)
        return torch.sum(x * alpha, dim=1) # (B, D)

class EmotionModel(nn.Module):
    def __init__(self, model_name_or_path, num_classes=4):
        super().__init__()
        self.backbone = WavLMModel.from_pretrained(model_name_or_path)
        hidden_size = self.backbone.config.hidden_size
        
        # Freeze feature encoder usually? User said "fully fine-tuned".
        # self.backbone.feature_extractor._freeze_parameters()
        
        self.pool = AttentionPooling(hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3) # VAD
        )
        
    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # (B, T, D)
        
        pooled = self.pool(sequence_output, attention_mask)
        
        logits = self.classifier(pooled)
        vad_output = self.regressor(pooled)
        
        return logits, vad_output
