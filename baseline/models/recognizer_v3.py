"""
Phase 3 Recognizer — Best of Phase 1 + Phase 2.

Key insight: BiLSTM beats Transformer on short sequences (W'=20).
Keep what works, add what helps:

  From Phase 1 (proven):
    - STN → ResNet-34 → BiLSTM → FC + CTC
    - OneCycleLR + weight_decay=1e-4

  From Phase 2 (improved fusion + SR regularization):
    - SpatialTemporalAttentionFusion (replaces simple AttentionFusion)
    - AuxSRBranch (training-only HR reconstruction → better backbone features)

Architecture:
    [STN] → [ResNet-34] → [SpatialTemporalAttentionFusion] → [BiLSTM] → [FC + CTC]
                              ↓ (training only)
                          [AuxSRBranch] → SR Loss (L1 + Perceptual)
"""

import torch
import torch.nn as nn

try:
    from .stn import STN, IdentitySTN
    from .backbone import ResNet34Backbone, VanillaCNN
    from .fusion import SpatialTemporalAttentionFusion
    from .sr_branch import AuxSRBranch
except ImportError:
    from stn import STN, IdentitySTN
    from backbone import ResNet34Backbone, VanillaCNN
    from fusion import SpatialTemporalAttentionFusion
    from sr_branch import AuxSRBranch


class Phase3Recognizer(nn.Module):
    """
    Phase 3: BiLSTM + SpatialTemporalFusion + SR Branch.

    Changes from Phase 1:
    - AttentionFusion → SpatialTemporalAttentionFusion (better multi-frame fusion)
    - Added AuxSRBranch (training-only HR reconstruction for backbone regularization)

    Changes from Phase 2:
    - TransformerEncoder → BiLSTM (BiLSTM proven better for short sequences)
    - weight_decay 0.05 → 1e-4
    - No backbone freeze — fine-tune everything from start
    """

    def __init__(
        self,
        num_classes,
        use_stn=True,
        use_resnet_backbone=True,
        # BiLSTM params (same as Phase 1)
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.25,
        # Fusion params
        fusion_reduction=16,
        # SR branch (training only)
        use_sr_branch=True,
        sr_target_h=43,
        sr_target_w=120,
    ):
        super().__init__()

        self.use_stn = use_stn
        self.use_resnet_backbone = use_resnet_backbone
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

        # Phase 3: SpatialTemporalAttentionFusion (from Phase 2)
        self.fusion = SpatialTemporalAttentionFusion(
            channels=backbone_channels,
            num_frames=5,
            reduction=fusion_reduction,
            dropout=0.1,
        )

        # Phase 3: Keep BiLSTM (proven better than Transformer for seq_len=20)
        self.rnn = nn.LSTM(
            backbone_channels,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Output classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # SR Branch (training only — reconstructs HR from backbone features)
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

        # SR Branch — reconstruct HR from mean backbone features (training only)
        sr_output = None
        if return_sr and self.sr_branch is not None:
            feat_for_sr = feat.view(b, t, *feat.shape[1:]).mean(dim=1)
            sr_output = self.sr_branch(feat_for_sr)

        # SpatialTemporal Fusion
        fused = self.fusion(feat)  # [B, 512, W']

        # BiLSTM sequence modeling
        fused = fused.permute(0, 2, 1)  # [B, W', 512]
        rnn_out, _ = self.rnn(fused)    # [B, W', hidden*2]

        # Classification
        out = self.fc(rnn_out)          # [B, W', num_classes]
        out = out.log_softmax(2)

        if return_sr:
            return out, sr_output
        return out

    def load_phase1_weights(self, phase1_path, device='cpu'):
        """
        Load Phase 1 checkpoint and transfer ALL compatible weights.

        Phase 1 has: stn, backbone, fusion (AttentionFusion), rnn, fc
        Phase 3 has: stn, backbone, fusion (SpatialTemporal), rnn, fc

        Transfer: stn, backbone, rnn, fc (same architecture)
        Skip: fusion (different module)
        """
        print(f"📦 Loading Phase 1 checkpoint: {phase1_path}")
        checkpoint = torch.load(phase1_path, map_location=device, weights_only=False)
        phase1_state = checkpoint['model_state_dict']

        # Only skip fusion (different architecture).
        # rnn + fc are IDENTICAL to Phase 1 → transfer them!
        SKIP_PREFIXES = ('fusion.',)

        transferred = []
        skipped = []
        model_state = self.state_dict()

        for name, param in phase1_state.items():
            if any(name.startswith(p) for p in SKIP_PREFIXES):
                skipped.append(f"{name} (new SpatialTemporalFusion)")
                continue
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name] = param
                    transferred.append(name)
                else:
                    skipped.append(f"{name} (shape: {param.shape} vs {model_state[name].shape})")
            else:
                skipped.append(f"{name} (not in Phase 3)")

        self.load_state_dict(model_state, strict=False)

        stn_n = sum(1 for n in transferred if 'stn' in n)
        bb_n  = sum(1 for n in transferred if 'backbone' in n)
        rnn_n = sum(1 for n in transferred if 'rnn' in n)
        fc_n  = sum(1 for n in transferred if 'fc' in n)

        print(f"   ✅ Transferred: {len(transferred)} parameters")
        print(f"   ⏭️  Skipped: {len(skipped)} parameters")
        print(f"   📊 STN: {stn_n} | Backbone: {bb_n} | RNN: {rnn_n} | FC: {fc_n}")
        print(f"   🆕 Random init: fusion (SpatialTemporalFusion), sr_branch")

        return checkpoint

    def get_model_info(self):
        return {
            'model': 'Phase3Recognizer',
            'stn': self.use_stn,
            'backbone': 'ResNet34' if self.use_resnet_backbone else 'VanillaCNN',
            'fusion': 'SpatialTemporalAttentionFusion',
            'sequence': 'BiLSTM',
            'sr_branch': self.use_sr_branch,
        }
