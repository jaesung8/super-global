import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerEncoder, TransformerEncoderLayer


class MatchERT(nn.Module):
    def __init__(self, device, d_global, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, normalize_before):
        super(MatchERT, self).__init__()
        assert (d_model % 2 == 0)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # self.pos_encoder = PositionEmbeddingSine(d_model//2, normalize=True, scale=2.0)
        self.remap = nn.Linear(d_global, d_model)
        self.scale_encoder = nn.Embedding(3, d_model)
        self.seg_encoder = nn.Embedding(4, d_model)
        self.classifier = nn.Linear(d_model, 1)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        self.device = device

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src_global, tgt_global):
        src_global = self.remap(src_global)
        tgt_global = self.remap(tgt_global)
        src_global = F.normalize(src_global, p=2, dim=-1)
        tgt_global = F.normalize(tgt_global, p=2, dim=-1)
        #     src_local  = F.normalize(src_local,  p=2, dim=-1)
        #     tgt_local  = F.normalize(tgt_local,  p=2, dim=-1)
        bsize, _, fsize = src_global.size()
        # src_scales = torch.Tensor([0, 1, 2]).to(self.device)
        # tgt_scales = torch.Tensor([0, 1, 2]).to(self.device)
        # src_scales = torch.permute(src_scales.expand(bsize, 1, 3), (0, 2, 1))
        # tgt_scales = torch.permute(tgt_scales.expand(bsize, 1, 3), (0, 2, 1))
        # src_global = src_global + self.scale_encoder(src_scales)
        # tgt_global = tgt_global + self.scale_encoder(tgt_scales)

        ##################################################################################################################
        cls_embed  = self.seg_encoder(src_global.new_zeros((bsize, 1), dtype=torch.long))
        sep_embed  = self.seg_encoder(src_global.new_ones((bsize, 1), dtype=torch.long))
        src_global = src_global + self.seg_encoder(2 * src_global.new_ones((bsize, 1), dtype=torch.long))
        tgt_global = tgt_global + self.seg_encoder(3 * src_global.new_ones((bsize, 1), dtype=torch.long))
        ##################################################################################################################
        input_feats = torch.cat([cls_embed, src_global, sep_embed, tgt_global], 1).permute(1,0,2)
        # input_mask = torch.cat([
        #     src_local.new_zeros((bsize, 2), dtype=torch.bool),
        #     src_mask,
        #     src_local.new_zeros((bsize, 2), dtype=torch.bool),
        #     tgt_mask
        # ], 1)
        logits = self.encoder(input_feats)
        logits = logits[0]
        return self.classifier(logits).view(-1)