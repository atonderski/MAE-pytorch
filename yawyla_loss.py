import torch

import torch.nn.functional as F
import einops


def compute_yawyla_weights(attn, weight_version):
    # Sum over the heads
    attn = torch.einsum('bhts->bts', attn)

    if weight_version == 'plain':
        # The weights are simply the attention
        weights = attn
    elif weight_version == 'bidirectional':
        # The weights are bidirectional attention, meaning if we look at something that looks back it is extra strong
        weights = (attn * einops.rearrange(attn, 'b t s -> b s t'))**0.5
    elif weight_version == 'attention-similarity':
        # The weights are the overall similarity between the whole attention maps
        weights = torch.einsum('bts,bsr->btr', attn, attn)
    else:
        raise ValueError(f"{weight_version} is not a recognized value")
    return weights

def yawyla_loss_func(enc_feats: torch.Tensor, enc_attn: torch.Tensor,
                     weight_version: str = 'plain', loss_version: str = 'plain'):
    # Detach! gradients should only flow through the target token.
    feats, attn = enc_feats.detach(), enc_attn.detach()

    weights = compute_yawyla_weights(attn, weight_version=weight_version)

    if loss_version == 'plain':
        # The target for a given token is the weighted average of all features
        target = torch.einsum('bsc,bts->btc', feats, weights)
        loss = F.mse_loss(enc_feats, target)
    elif loss_version == 'pooled':
        # First we construct a pooled feature for each token
        pooled = torch.einsum('bsc,bts->btc', feats, weights)
        # Then we assign the target as the weighted average of pooled tokens
        target = torch.einsum('bsc,bts->btc', pooled, weights)
        # The loss is simply MSE
        loss = F.mse_loss(enc_feats, target)
    elif loss_version == 'contrast':
        # First we construct a pooled feature for each token
        pooled = torch.einsum('bsc,bts->btc', feats, weights)
        # Then we apply a smooth contrastive loss based on these pooled segmentations
        loss = None
        raise NotImplementedError()
    else:
        raise ValueError(f"{loss_version} is not a recognized value")

    return loss
