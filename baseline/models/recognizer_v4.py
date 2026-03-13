"""
Phase 5 Recognizer — CTC + Attention Dual Decoder.

Key insight: Adding an attention decoder alongside CTC provides:
  1. Character-level dependency modeling (CTC assumes conditional independence)
  2. Implicit language model through autoregressive decoding
  3. Better encoder regularization from multi-task learning
  4. Estimated +1-2% accuracy improvement

Architecture:
    [STN] → [ResNet-34] → [SpatialTemporalFusion] → [BiLSTM] ──┬── [CTC Head] → CTC Loss
                                                                  └── [Attention Decoder] → CE Loss
                              ↓ (training only)
                          [AuxSRBranch] → SR Loss

During training:  loss = λ_ctc * CTC_loss + λ_attn * Attention_CE_loss + λ_sr * SR_loss
During inference: CTC greedy/beam decode (attention decoder not needed at inference)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ════════════════════════════════════════════════════════════════════════════
# Attention Decoder
# ════════════════════════════════════════════════════════════════════════════
class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over encoder outputs."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden, enc_proj=None):
        """
        Args:
            encoder_out:    [B, T, encoder_dim]    — BiLSTM outputs
            decoder_hidden: [B, decoder_dim]        — GRU hidden state
            enc_proj:       [B, T, attention_dim]   — precomputed W_enc @ encoder_out
        Returns:
            context:        [B, encoder_dim]
            attn_weights:   [B, T]
        """
        if enc_proj is None:
            enc_proj = self.W_enc(encoder_out)   # [B, T, attn_dim]

        dec_proj = self.W_dec(decoder_hidden)    # [B, attn_dim]
        dec_proj = dec_proj.unsqueeze(1)         # [B, 1, attn_dim]

        energy = self.v(torch.tanh(enc_proj + dec_proj))  # [B, T, 1]
        attn_weights = F.softmax(energy.squeeze(2), dim=1)  # [B, T]

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out)  # [B, 1, enc_dim]
        context = context.squeeze(1)  # [B, enc_dim]

        return context, attn_weights


class AttentionDecoder(nn.Module):
    """
    Autoregressive attention decoder for character sequence prediction.

    Uses Bahdanau attention over BiLSTM encoder outputs.
    Teacher forcing during training, greedy during inference.

    Architecture:
        Embedding(prev_char) + Context → GRUCell → FC → logits
    """

    SOS_IDX = 0  # Use CTC blank as SOS token
    MAX_DECODE_LEN = 10  # Brazilian plates: 7 chars + margin

    def __init__(
        self,
        encoder_dim,       # BiLSTM output dim (hidden_size * 2 = 512)
        decoder_dim=256,   # GRU hidden size
        attention_dim=256,  # Attention bottleneck
        embed_dim=64,      # Character embedding dim
        num_classes=37,    # 0=blank + 36 chars
        max_decode_len=10,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.num_classes = num_classes
        self.max_decode_len = max_decode_len

        # Character embedding
        self.embedding = nn.Embedding(num_classes, embed_dim, padding_idx=0)

        # Attention
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        # GRU decoder cell
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)

        # Output projection: [decoder_hidden + context] → logits
        self.fc_out = nn.Sequential(
            nn.Linear(decoder_dim + encoder_dim, decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, num_classes),
        )

        # Initialize decoder hidden from encoder
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, targets=None, teacher_forcing_ratio=0.5):
        """
        Args:
            encoder_outputs:       [B, T, encoder_dim] from BiLSTM
            targets:               [B, max_len] padded target indices (for teacher forcing)
            teacher_forcing_ratio: probability of using ground truth at each step

        Returns:
            logits: [B, max_decode_len, num_classes]
        """
        B = encoder_outputs.size(0)
        device = encoder_outputs.device

        # Initialize decoder hidden state from mean-pooled encoder
        h = self.init_h(encoder_outputs.mean(dim=1))  # [B, decoder_dim]
        h = torch.tanh(h)

        # Precompute attention projection (saves computation)
        enc_proj = self.attention.W_enc(encoder_outputs)  # [B, T, attn_dim]

        # Start token (blank/SOS)
        y = torch.full((B,), self.SOS_IDX, dtype=torch.long, device=device)

        outputs = []
        for t in range(self.max_decode_len):
            # Embed previous prediction
            embed = self.embedding(y)       # [B, embed_dim]
            embed = self.dropout(embed)

            # Attention
            context, _ = self.attention(encoder_outputs, h, enc_proj)  # [B, enc_dim]

            # GRU step
            gru_input = torch.cat([embed, context], dim=1)  # [B, embed_dim + enc_dim]
            h = self.gru(gru_input, h)                      # [B, decoder_dim]

            # Output prediction
            out = self.fc_out(torch.cat([h, context], dim=1))  # [B, num_classes]
            outputs.append(out)

            # Teacher forcing
            if targets is not None and random.random() < teacher_forcing_ratio:
                y = targets[:, t] if t < targets.size(1) else torch.zeros(B, dtype=torch.long, device=device)
            else:
                y = out.argmax(dim=1)

        return torch.stack(outputs, dim=1)  # [B, max_decode_len, num_classes]


# ════════════════════════════════════════════════════════════════════════════
# Phase 5 Recognizer
# ════════════════════════════════════════════════════════════════════════════
class Phase5Recognizer(nn.Module):
    """
    Phase 5: CTC + Attention Dual Decoder.

    Same encoder as Phase 3/4 (STN → ResNet34 → SpatialTemporalFusion → BiLSTM)
    with two decoder heads:
      1. CTC head: FC → LogSoftmax (alignment-free, fast inference)
      2. Attention head: GRU + Bahdanau attention (character dependencies, regularization)
    """

    def __init__(
        self,
        num_classes,
        use_stn=True,
        use_resnet_backbone=True,
        # BiLSTM params
        hidden_size=256,
        num_lstm_layers=2,
        dropout=0.25,
        # Fusion params
        fusion_reduction=16,
        # Attention decoder params
        attention_dim=256,
        decoder_dim=256,
        embed_dim=64,
        max_decode_len=10,
        attn_dropout=0.2,
        # SR branch
        use_sr_branch=True,
        sr_target_h=43,
        sr_target_w=120,
    ):
        super().__init__()

        self.use_stn = use_stn
        self.use_resnet_backbone = use_resnet_backbone
        self.use_sr_branch = use_sr_branch

        # ── Encoder (same as Phase 3/4) ──────────────────────────────────

        # STN
        if use_stn:
            self.stn = STN(input_channels=3)
        else:
            self.stn = IdentitySTN()

        # Backbone
        if use_resnet_backbone:
            self.backbone = ResNet34Backbone(pretrained=True)
        else:
            self.backbone = VanillaCNN()

        backbone_channels = self.backbone.output_channels  # 512

        # Multi-frame fusion
        self.fusion = SpatialTemporalAttentionFusion(
            channels=backbone_channels,
            num_frames=5,
            reduction=fusion_reduction,
            dropout=0.1,
        )

        # BiLSTM sequence encoder
        self.rnn = nn.LSTM(
            backbone_channels,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        encoder_dim = hidden_size * 2  # 512

        # ── CTC Head (same as Phase 3/4) ─────────────────────────────────
        self.ctc_fc = nn.Linear(encoder_dim, num_classes)

        # ── Attention Decoder Head (NEW) ─────────────────────────────────
        self.attention_decoder = AttentionDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
            max_decode_len=max_decode_len,
            dropout=attn_dropout,
        )

        # ── SR Branch ────────────────────────────────────────────────────
        if use_sr_branch:
            self.sr_branch = AuxSRBranch(
                in_channels=backbone_channels,
                target_h=sr_target_h,
                target_w=sr_target_w,
            )
        else:
            self.sr_branch = None

    def forward(self, x, targets=None, return_sr=False, return_attention=False,
                teacher_forcing_ratio=0.5):
        """
        Args:
            x:         [B, T, C, H, W] — T=5 frames
            targets:   [B, max_decode_len] — padded target indices (for attention decoder)
            return_sr: return SR reconstruction
            return_attention: return attention decoder output

        Returns (depending on flags):
            ctc_log_probs: [B, W', num_classes]
            attn_logits:   [B, max_decode_len, num_classes]   (if return_attention)
            sr_output:     [B, 3, sr_H, sr_W]                 (if return_sr)
        """
        b, t, c, h, w = x.size()

        # ── Encoder ──────────────────────────────────────────────────────

        # STN per frame
        x = x.view(b * t, c, h, w)
        x = self.stn(x)  # [B*T, C, H, W]

        # Backbone features
        feat = self.backbone(x)  # [B*T, 512, H', W']

        # SR Branch (training only)
        sr_output = None
        if return_sr and self.sr_branch is not None:
            feat_for_sr = feat.view(b, t, *feat.shape[1:]).mean(dim=1)
            sr_output = self.sr_branch(feat_for_sr)

        # Multi-frame fusion
        fused = self.fusion(feat)  # [B, 512, W']

        # BiLSTM
        fused = fused.permute(0, 2, 1)  # [B, W', 512]
        rnn_out, _ = self.rnn(fused)     # [B, W', hidden*2 = 512]

        # ── CTC Head ────────────────────────────────────────────────────
        ctc_out = self.ctc_fc(rnn_out)       # [B, W', num_classes]
        ctc_log_probs = ctc_out.log_softmax(2)

        # ── Attention Decoder Head ───────────────────────────────────────
        attn_logits = None
        if return_attention:
            attn_logits = self.attention_decoder(
                rnn_out.detach() if not self.training else rnn_out,
                targets=targets,
                teacher_forcing_ratio=teacher_forcing_ratio if self.training else 0.0,
            )

        # ── Return ──────────────────────────────────────────────────────
        result = [ctc_log_probs]
        if return_attention:
            result.append(attn_logits)
        if return_sr:
            result.append(sr_output)

        return result[0] if len(result) == 1 else tuple(result)

    def load_phase4_weights(self, phase4_path, device='cpu'):
        """
        Load Phase 4 checkpoint (Phase3Recognizer) and transfer encoder weights.

        Phase 4 has:  stn, backbone, fusion, rnn, fc, sr_branch
        Phase 5 has:  stn, backbone, fusion, rnn, ctc_fc, attention_decoder, sr_branch

        Transfer:     stn, backbone, fusion, rnn, sr_branch (identical)
                      fc → ctc_fc (same linear layer, renamed)
        New (random): attention_decoder (brand new module)
        """
        print(f"Loading Phase 4 checkpoint: {phase4_path}")
        checkpoint = torch.load(phase4_path, map_location=device, weights_only=False)
        phase4_state = checkpoint['model_state_dict']

        transferred = []
        skipped = []
        model_state = self.state_dict()

        for name, param in phase4_state.items():
            # Rename fc → ctc_fc
            target_name = name
            if name.startswith('fc.'):
                target_name = 'ctc_' + name  # fc.weight → ctc_fc.weight

            if target_name in model_state:
                if model_state[target_name].shape == param.shape:
                    model_state[target_name] = param
                    transferred.append(f"{name} → {target_name}")
                else:
                    skipped.append(f"{name} (shape: {param.shape} vs {model_state[target_name].shape})")
            else:
                skipped.append(f"{name} (not in Phase 5)")

        self.load_state_dict(model_state, strict=False)

        # Count by module
        stn_n = sum(1 for n in transferred if 'stn' in n)
        bb_n  = sum(1 for n in transferred if 'backbone' in n)
        fus_n = sum(1 for n in transferred if 'fusion' in n)
        rnn_n = sum(1 for n in transferred if 'rnn' in n)
        fc_n  = sum(1 for n in transferred if 'fc' in n)
        sr_n  = sum(1 for n in transferred if 'sr_branch' in n)

        print(f"   Transferred: {len(transferred)} parameters")
        print(f"   Skipped:     {len(skipped)} parameters")
        print(f"   STN: {stn_n} | Backbone: {bb_n} | Fusion: {fus_n} | "
              f"RNN: {rnn_n} | CTC FC: {fc_n} | SR: {sr_n}")
        print(f"   Random init: attention_decoder (NEW)")

        # Count new parameters
        new_params = sum(p.numel() for n, p in self.named_parameters()
                        if 'attention_decoder' in n)
        print(f"   New attention decoder params: {new_params:,}")

        return checkpoint

    def get_model_info(self):
        attn_params = sum(p.numel() for n, p in self.named_parameters()
                         if 'attention_decoder' in n)
        total_params = sum(p.numel() for p in self.parameters())
        return {
            'model': 'Phase5Recognizer',
            'stn': self.use_stn,
            'backbone': 'ResNet34' if self.use_resnet_backbone else 'VanillaCNN',
            'fusion': 'SpatialTemporalAttentionFusion',
            'sequence': 'BiLSTM',
            'ctc_head': True,
            'attention_decoder': True,
            'sr_branch': self.use_sr_branch,
            'attention_params': attn_params,
            'total_params': total_params,
        }
