"""
Phase 7 Recognizer — Larger architecture for breaking 78.47% ceiling.

vs Phase3Recognizer (Phase 4 v2 best = 78.47%):
  - ResNet50 backbone (was ResNet34): deeper features, Bottleneck blocks
  - BiLSTM hidden=512 (was 256): 2× more sequence modeling capacity
  - FC: 1024→37 (was 512→37)
  - ~30M params (was 15.5M)

Everything else identical:
  - STN for geometric rectification
  - SpatialTemporalAttentionFusion for multi-frame fusion
  - AuxSRBranch for training-only HR reconstruction
  - CTC output with log_softmax

Architecture:
    [STN] → [ResNet-50] → [SpatialTemporalFusion] → [BiLSTM(512×2)] → [FC(1024→37)]
                              ↓ (training only)
                          [AuxSRBranch] → SR Loss
"""

import torch
import torch.nn as nn

try:
    from .stn import STN, IdentitySTN
    from .backbone import ResNet34Backbone
    from .backbone_v2 import ResNet50Backbone
    from .fusion import SpatialTemporalAttentionFusion
    from .sr_branch import AuxSRBranch
except ImportError:
    from stn import STN, IdentitySTN
    from backbone import ResNet34Backbone
    from backbone_v2 import ResNet50Backbone
    from fusion import SpatialTemporalAttentionFusion
    from sr_branch import AuxSRBranch


class Phase7Recognizer(nn.Module):
    """
    Phase 7: Larger architecture — ResNet50 + BiLSTM(512).

    Key upgrade: more model capacity to distinguish confusing character pairs
    (8↔6, V↔Y, 5↔6, 9↔5, M↔H, Q↔O, D↔B).
    """

    def __init__(
        self,
        num_classes,
        use_stn=True,
        backbone_type='resnet50',       # 'resnet34' or 'resnet50'
        # BiLSTM params
        hidden_size=512,                # doubled from 256
        num_lstm_layers=2,
        dropout=0.3,                    # slightly higher for larger model
        # Fusion params
        fusion_reduction=16,
        # SR branch (training only)
        use_sr_branch=True,
        sr_target_h=43,
        sr_target_w=120,
    ):
        super().__init__()

        self.use_stn = use_stn
        self.backbone_type = backbone_type
        self.use_sr_branch = use_sr_branch

        # STN
        if use_stn:
            self.stn = STN(input_channels=3)
        else:
            self.stn = IdentitySTN()

        # Backbone — ResNet50 or ResNet34
        if backbone_type == 'resnet50':
            self.backbone = ResNet50Backbone(pretrained=True)
        else:
            self.backbone = ResNet34Backbone(pretrained=True)

        backbone_channels = self.backbone.output_channels  # 512

        # SpatialTemporalAttentionFusion (same as Phase 3)
        self.fusion = SpatialTemporalAttentionFusion(
            channels=backbone_channels,
            num_frames=5,
            reduction=fusion_reduction,
            dropout=0.1,
        )

        # BiLSTM — larger hidden size
        self.rnn = nn.LSTM(
            backbone_channels,            # 512 input
            hidden_size,                  # 512 hidden (was 256)
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Dropout before FC for regularization
        self.fc_dropout = nn.Dropout(dropout)

        # Output classifier: hidden*2 (bidirectional) → num_classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # SR Branch (training only)
        if use_sr_branch:
            self.sr_branch = AuxSRBranch(
                in_channels=backbone_channels,
                target_h=sr_target_h,
                target_w=sr_target_w,
            )
        else:
            self.sr_branch = None

    def forward(self, x, return_sr=False):
        """
        Args:
            x:         [B, T, C, H, W] where T=5 frames
            return_sr: Return SR reconstruction (training only)

        Returns:
            log_probs: [B, W', num_classes]
            sr_output: [B, 3, sr_H, sr_W] (only if return_sr=True)
        """
        b, t, c, h, w = x.size()

        # STN: rectify each frame
        x = x.view(b * t, c, h, w)
        x = self.stn(x)                    # [B*T, C, H, W]

        # Backbone: extract features
        feat = self.backbone(x)             # [B*T, 512, H', W']

        # SR Branch (training only)
        sr_output = None
        if return_sr and self.sr_branch is not None:
            feat_for_sr = feat.view(b, t, *feat.shape[1:]).mean(dim=1)
            sr_output = self.sr_branch(feat_for_sr)

        # Multi-frame fusion
        fused = self.fusion(feat)           # [B, 512, W']

        # BiLSTM sequence modeling
        fused = fused.permute(0, 2, 1)      # [B, W', 512]
        rnn_out, _ = self.rnn(fused)        # [B, W', 1024]

        # Classification with dropout
        out = self.fc_dropout(rnn_out)
        out = self.fc(out)                  # [B, W', num_classes]
        out = out.log_softmax(2)

        if return_sr:
            return out, sr_output
        return out

    def get_model_info(self):
        return {
            'model': 'Phase7Recognizer',
            'stn': self.use_stn,
            'backbone': self.backbone_type.upper(),
            'fusion': 'SpatialTemporalAttentionFusion',
            'sequence': 'BiLSTM',
            'hidden_size': self.rnn.hidden_size,
            'sr_branch': self.use_sr_branch,
        }
