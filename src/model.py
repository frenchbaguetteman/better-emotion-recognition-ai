import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig, AutoModel, AutoConfig
from config import Config

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls = nn.Linear(in_dim, 1)
        
    def forward(self, x, attention_mask=None):
        # x: (B, T, D)
        scores = self.cls(x) # (B, T, 1)
        
        if attention_mask is not None:
            # attention_mask: (B, T_orig)
            # x: (B, T_feat, D)
            B, T_feat, D = x.shape
            
            # If mask length mismatches (due to downsampling in WavLM), interpolate it
            if attention_mask.shape[1] != T_feat:
                # Interpolate mask to match feature length
                # shape: (B, 1, T_orig) -> (B, 1, T_feat)
                mask_float = attention_mask.unsqueeze(1).float()
                mask_interp = torch.nn.functional.interpolate(
                    mask_float, size=T_feat, mode='nearest'
                ) # (B, 1, T_feat)
                mask = mask_interp.transpose(1, 2) # (B, T_feat, 1)
            else:
                mask = attention_mask.unsqueeze(-1) # (B, T, 1)
                
            # Use -1e4 instead of -1e9 to avoid Float16 overflow
            scores = scores.masked_fill(mask == 0, -1e4)
            
        alpha = torch.softmax(scores, dim=1) # (B, T, 1)
        return torch.sum(x * alpha, dim=1) # (B, D)

class EmotionModel(nn.Module):
    def __init__(self, audio_model_name=None, text_model_name=Config.TEXT_MODEL_NAME, num_classes=4, use_text=True, model_name_or_path=None):
        super().__init__()
        # Handle backward compatibility
        if model_name_or_path is not None:
            audio_model_name = model_name_or_path
        if audio_model_name is None:
            audio_model_name = "microsoft/wavlm-large"
        
        self.use_text = use_text
        
        # Audio Backbone
        self.backbone = WavLMModel.from_pretrained(audio_model_name, apply_spec_augment=True if 'wavlm' in audio_model_name else False)
        hidden_size = self.backbone.config.hidden_size
        self.audio_norm = nn.Identity()
        
        # Text Backbone
        if use_text:
            self.text_backbone = AutoModel.from_pretrained(text_model_name)
            text_hidden_size = self.text_backbone.config.hidden_size
            self.text_norm = nn.Identity()
            self.text_dropout = Config.TEXT_DROPOUT
        else:
            self.text_backbone = None
            text_hidden_size = 0
            self.text_norm = None
            self.text_dropout = 0.0
        
        # Pooling
        self.audio_pool = AttentionPooling(hidden_size)
        
        # Classifier
        combined_dim = hidden_size + text_hidden_size
        if combined_dim == hidden_size:
            # Audio only
            combined_dim = hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2), # Increased dropout for fusion
            nn.Linear(512, num_classes)
        )
        
        # VAD Regressor (Arousal/Valence/Dominance if available, but here used as dummy or aux task)
        # We'll map it from combined features too for better consistency
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3) 
        )
        
    def forward(self, input_values, attention_mask=None, input_ids=None, text_attention_mask=None):
        # 1. Audio Forward
        audio_outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        audio_seq = audio_outputs.last_hidden_state # (B, T_audio, D_audio)
        audio_emb = self.audio_pool(audio_seq, attention_mask) # (B, D_audio)
        audio_emb = self.audio_norm(audio_emb)
        
        # 2. Text Forward (if enabled)
        if self.use_text and input_ids is not None:
            text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=text_attention_mask)
            text_hidden = text_outputs.last_hidden_state
            if text_attention_mask is None:
                text_emb = text_hidden[:, 0, :] # (B, D_text)
            else:
                mask = text_attention_mask.unsqueeze(-1).float()
                masked = text_hidden * mask
                denom = mask.sum(dim=1).clamp(min=1.0)
                text_emb = masked.sum(dim=1) / denom
            
            text_emb = self.text_norm(text_emb)
            
            if self.training and self.text_dropout > 0:
                drop_mask = torch.rand(text_emb.size(0), 1, device=text_emb.device) < self.text_dropout
                text_emb = text_emb.masked_fill(drop_mask, 0.0)
            
            if text_attention_mask is not None:
                has_text = (text_attention_mask.sum(dim=1, keepdim=True) > 0).float()
                text_emb = text_emb * has_text
            
            combined = torch.cat([audio_emb, text_emb], dim=1)
        elif self.use_text:
            # Text model but no text given: use a zero text embedding (same as an empty transcript)
            text_emb = audio_emb.new_zeros(audio_emb.size(0), self.text_backbone.config.hidden_size)
            combined = torch.cat([audio_emb, text_emb], dim=1)
        else:
            # Audio only
            combined = audio_emb
        
        logits = self.classifier(combined)
        vad_output = self.regressor(combined)
        
        return logits, vad_output
