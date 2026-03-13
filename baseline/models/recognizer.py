"""
Phase 1 Recognizer - Enhanced Multi-Frame CRNN with STN + ResNet-34.

Upgrades from baseline:
- Optional STN for spatial alignment
- ResNet-34 backbone (pretrained) instead of vanilla CNN
- Keep AttentionFusion and BiLSTM (will upgrade in Phase 2)
"""

import torch
import torch.nn as nn

try:
    from .stn import STN, IdentitySTN
    from .backbone import ResNet34Backbone, VanillaCNN
    from .fusion import AttentionFusion
except ImportError:
    from stn import STN, IdentitySTN
    from backbone import ResNet34Backbone, VanillaCNN
    from fusion import AttentionFusion


class Phase1Recognizer(nn.Module):
    """
    Phase 1: Multi-Frame License Plate Recognizer.
    
    Architecture:
        [STN] → [ResNet-34 Backbone] → [AttentionFusion] → [BiLSTM] → [FC + CTC]
    
    Changes from baseline MultiFrameCRNN:
    - Added STN module for geometric alignment
    - Replaced vanilla CNN with ResNet-34 backbone
    - Kept AttentionFusion and BiLSTM (Phase 2 will upgrade these)
    """
    
    def __init__(
        self, 
        num_classes,
        use_stn=True,
        use_resnet_backbone=True,
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.25
    ):
        super().__init__()
        
        self.use_stn = use_stn
        self.use_resnet_backbone = use_resnet_backbone
        
        # Phase 1.1: Spatial Transformer Network (optional)
        if use_stn:
            self.stn = STN(input_channels=3)
        else:
            self.stn = IdentitySTN()
        
        # Phase 1.2: Backbone (ResNet-34 or vanilla CNN)
        if use_resnet_backbone:
            self.backbone = ResNet34Backbone(pretrained=True)
        else:
            self.backbone = VanillaCNN()
        
        backbone_channels = self.backbone.output_channels  # 512
        
        # Phase 1: Keep original AttentionFusion (will upgrade in Phase 2)
        self.fusion = AttentionFusion(channels=backbone_channels)
        
        # Phase 1: Keep BiLSTM (will replace with Transformer in Phase 2)
        self.rnn = nn.LSTM(
            backbone_channels, 
            hidden_size, 
            num_layers=num_lstm_layers,
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, T, C, H, W] where T=5 frames
        
        Returns:
            Log probabilities [B, W', num_classes] for CTC loss
        """
        b, t, c, h, w = x.size()
        
        # Process each frame through STN
        x = x.view(b * t, c, h, w)
        x = self.stn(x)  # [B*T, C, H, W]
        
        # Extract features through backbone
        feat = self.backbone(x)  # [B*T, 512, H', W']
        
        # Temporal fusion across frames
        fused = self.fusion(feat)  # [B, 512, W']
        
        # RNN sequence modeling
        fused = fused.permute(0, 2, 1)  # [B, W', 512]
        rnn_out, _ = self.rnn(fused)  # [B, W', hidden*2]
        
        # Classification
        out = self.fc(rnn_out)  # [B, W', num_classes]
        out = out.log_softmax(2)
        
        return out
    
    def get_model_info(self):
        """Return model configuration info"""
        return {
            'model': 'Phase1Recognizer',
            'stn': self.use_stn,
            'backbone': 'ResNet34' if self.use_resnet_backbone else 'VanillaCNN',
            'fusion': 'AttentionFusion',
            'sequence': 'BiLSTM'
        }
