"""
Phase 2 Recognizer - Full Architecture Upgrade.

Upgrades from Phase 1:
- 2.1 SpatialTemporalAttentionFusion: 3-stage (Channel→Spatial→Temporal) fusion
- 2.2 Lighter TransformerEncoder (2 layers, 4 heads) replaces BiLSTM
- 2.3 Auxiliary SR Branch: forces backbone to learn HR features (training only)
- Compatible with Phase 1 checkpoint (partial loading: STN + backbone)
"""

import torch
import torch.nn as nn

try:
    from .stn import STN, IdentitySTN
    from .backbone import ResNet34Backbone, VanillaCNN
    from .fusion import SpatialTemporalAttentionFusion
    from .transformer import TransformerSequenceEncoder
    from .sr_branch import AuxSRBranch
except ImportError:
    from stn import STN, IdentitySTN
    from backbone import ResNet34Backbone, VanillaCNN
    from fusion import SpatialTemporalAttentionFusion
    from transformer import TransformerSequenceEncoder
    from sr_branch import AuxSRBranch


class Phase2Recognizer(nn.Module):
    """
    Phase 2: Multi-Frame License Plate Recognizer with Spatial-Temporal Fusion.
    
    Architecture:
        [STN] → [ResNet-34] → [SpatialTemporalAttentionFusion] → [TransformerEncoder] → [FC + CTC]
    
    Changes from Phase 1:
    - 2.1 AttentionFusion → SpatialTemporalAttentionFusion
    - 2.2 BiLSTM → Lightweight TransformerEncoder (2 layers, 4 heads)
    - 2.3 AuxSRBranch for training-time feature regularization
    """
    
    def __init__(
        self,
        num_classes,
        use_stn=True,
        use_resnet_backbone=True,
        # Transformer params (lightweight: 2L/4H beats BiLSTM on small datasets)
        d_model=512,
        nhead=4,
        num_transformer_layers=2,
        dim_feedforward=1024,
        # Fusion params (SE reduction ratio)
        fusion_reduction=16,
        # Regularization
        dropout=0.1,
        # SR branch (training only)
        use_sr_branch=True,
        sr_target_h=43,
        sr_target_w=120,
    ):
        super().__init__()
        
        self.use_stn = use_stn
        self.use_resnet_backbone = use_resnet_backbone
        self.d_model = d_model
        self.use_sr_branch = use_sr_branch
        
        # STN (same as Phase 1)
        if use_stn:
            self.stn = STN(input_channels=3)
        else:
            self.stn = IdentitySTN()
        
        # Backbone (same as Phase 1)
        if use_resnet_backbone:
            self.backbone = ResNet34Backbone(pretrained=True)
        else:
            self.backbone = VanillaCNN()
        
        backbone_channels = self.backbone.output_channels  # 512
        
        # Phase 2: Spatial-Temporal Attention Fusion (3-stage)
        self.fusion = SpatialTemporalAttentionFusion(
            channels=backbone_channels,
            num_frames=5,
            reduction=fusion_reduction,
            dropout=dropout
        )
        
        # Feature projection (if backbone channels != d_model)
        self.feature_proj = None
        if backbone_channels != d_model:
            self.feature_proj = nn.Sequential(
                nn.Linear(backbone_channels, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
        
        # Feature dropout before sequence modeling
        self.feat_dropout = nn.Dropout(dropout)
        
        # Phase 2: Transformer Encoder (replaces BiLSTM)
        self.sequence_model = TransformerSequenceEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output classifier
        self.fc = nn.Linear(d_model, num_classes)
        
        # 2.3: Auxiliary SR Branch (training only — reconstructs HR from backbone features)
        if use_sr_branch:
            self.sr_branch = AuxSRBranch(
                in_channels=backbone_channels,
                target_h=sr_target_h,
                target_w=sr_target_w
            )
        else:
            self.sr_branch = None
    
    def forward(self, x, return_sr=False):
        """
        Args:
            x:         Input tensor [B, T, C, H, W] where T=5 frames
            return_sr: If True, also return SR reconstruction (training only)
        
        Returns:
            log_probs: [B, W', num_classes] for CTC loss
            sr_output: [B, 3, sr_H, sr_W] only when return_sr=True
        """
        b, t, c, h, w = x.size()
        
        # Process each frame through STN
        x = x.view(b * t, c, h, w)
        x = self.stn(x)  # [B*T, C, H, W]
        
        # Extract features through backbone
        feat = self.backbone(x)  # [B*T, 512, H', W']
        
        # 2.3: SR Branch — reconstruct HR from mean backbone features (training only)
        sr_output = None
        if return_sr and self.sr_branch is not None:
            feat_for_sr = feat.view(b, t, *feat.shape[1:]).mean(dim=1)  # [B, 512, H', W']
            sr_output = self.sr_branch(feat_for_sr)  # [B, 3, sr_H, sr_W]
        
        # 2.1: Spatial-Temporal Attention Fusion
        fused = self.fusion(feat)  # [B, 512, W']
        
        # Reshape for sequence modeling: [B, W', 512]
        fused = fused.permute(0, 2, 1)  # [B, W', 512]
        
        # Project features if needed
        if self.feature_proj is not None:
            fused = self.feature_proj(fused)
        
        fused = self.feat_dropout(fused)
        
        # 2.2: Transformer sequence modeling
        seq_out = self.sequence_model(fused)  # [B, W', d_model]
        
        # Classification
        out = self.fc(seq_out)  # [B, W', num_classes]
        out = out.log_softmax(2)
        
        if return_sr:
            return out, sr_output
        return out
    
    def load_phase1_weights(self, phase1_path, device='cpu'):
        """
        Load Phase 1 checkpoint and transfer compatible weights.
        
        Transfers: STN, backbone weights (frozen initially)
        Skips: fusion, rnn, fc (new architecture — different feature distribution)
        """
        print(f"📦 Loading Phase 1 checkpoint: {phase1_path}")
        checkpoint = torch.load(phase1_path, map_location=device, weights_only=False)
        phase1_state = checkpoint['model_state_dict']
        
        # Only transfer STN + backbone; skip fc (BiLSTM→Transformer feature mismatch)
        SKIP_PREFIXES = ('fusion.', 'rnn.', 'fc.')
        
        transferred = []
        skipped = []
        
        model_state = self.state_dict()
        
        for name, param in phase1_state.items():
            if any(name.startswith(p) for p in SKIP_PREFIXES):
                skipped.append(f"{name} (Phase 2 re-init)")
                continue
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred.append(name)
                else:
                    skipped.append(f"{name} (shape mismatch: {param.shape} vs {model_state[name].shape})")
            else:
                skipped.append(f"{name} (not in Phase 2)")
        
        self.load_state_dict(model_state, strict=False)
        
        print(f"   ✅ Transferred: {len(transferred)} parameters")
        print(f"   ⏭️  Skipped: {len(skipped)} parameters")
        
        # Log key transfers
        stn_count = sum(1 for n in transferred if 'stn' in n)
        backbone_count = sum(1 for n in transferred if 'backbone' in n)
        print(f"   📊 STN: {stn_count} params, Backbone: {backbone_count} params")
        
        return checkpoint
    
    def get_model_info(self):
        """Return model configuration info"""
        return {
            'model': 'Phase2Recognizer',
            'stn': self.use_stn,
            'backbone': 'ResNet34' if self.use_resnet_backbone else 'VanillaCNN',
            'fusion': 'SpatialTemporalAttentionFusion',
            'sequence': 'TransformerEncoder',
            'd_model': self.d_model,
            'sr_branch': self.use_sr_branch,
            'nhead': 4,
            'num_transformer_layers': 2,
            'fusion_reduction': 16,
        }
