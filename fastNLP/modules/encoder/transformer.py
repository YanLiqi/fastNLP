from torch import nn
import torch
from ..aggregator.attention import MultiHeadAtte
from ..other_modules import LayerNormalization


def sequence_mask(seq):
    seq = seq.detach().numpy()
    shape = seq.shape
    batch_size = shape[0]
    seq_len = shape[1]
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class TransformerEncoder(nn.Module):
    class SubLayer(nn.Module):
        def __init__(self, input_size, output_size, key_size, value_size, num_atte):
            super(TransformerEncoder.SubLayer, self).__init__()
            self.atte = MultiHeadAtte(input_size, output_size, key_size, value_size, num_atte)
            self.norm1 = LayerNormalization(output_size)
            self.ffn = nn.Sequential(nn.Linear(output_size, output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size, output_size))
            self.norm2 = LayerNormalization(output_size)

        def forward(self, input, seq_mask=None):
            attention = self.atte(input, input, input)
            norm_atte = self.norm1(attention + input)
            output = self.ffn(norm_atte)
            return self.norm2(output + norm_atte)

    def __init__(self, num_layers, **kwargs):
        super(TransformerEncoder, self).__init__()
        # self.layers = nn.Sequential(*[self.SubLayer(**kwargs) for _ in range(num_layers)])
        self.encoder_layers = nn.ModuleList([self.SubLayer(**kwargs) for _ in range(num_layers)])

    def forward(self, x, seq_mask=None):
        # return self.layers(x, seq_mask)
        seq_mask = sequence_mask(x)
        for encoder in self.encoder_layers:
            y = encoder(x, seq_mask)

        return y
