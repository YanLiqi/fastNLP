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
                    diagonal=0)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class TransformerDecoder(nn.Module):
    class SubLayer(nn.Module):
        def __init__(self, input_size, output_size, key_size, value_size, num_atte):
            super(TransformerDecoder.SubLayer, self).__init__()
            self.atte1 = MultiHeadAtte(input_size, output_size, key_size, value_size, num_atte)
            self.norm1 = LayerNormalization(output_size)
            self.atte2 = MultiHeadAtte(input_size, output_size, key_size, value_size, num_atte)
            self.norm2 = LayerNormalization(output_size)
            self.ffn = nn.Sequential(nn.Linear(output_size, output_size),
                                     nn.ReLU(),
                                     nn.Linear(output_size, output_size))
            self.norm3 = LayerNormalization(output_size)

        def forward(self, y, x, seq_mask=None):
            attention1 = self.atte1(y, y, y)
            norm_atte1 = self.norm1(attention1 + y)
            attention2 = self.atte2(y, x, x, seq_mask)
            norm_atte2 = self.norm2(attention2 + norm_atte1)
            output = self.ffn(norm_atte2)
            return self.norm2(output + norm_atte2)

    def __init__(self, num_layers, **kwargs):
        super(TransformerDecoder, self).__init__()
        # self.layers = nn.Sequential(*[self.SubLayer(**kwargs) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([self.SubLayer(**kwargs) for _ in range(num_layers)])

    def forward(self, y, x, seq_mask=None):
        # return self.layers(y, x=x, seq_mask=seq_mask)
        seq_mask = sequence_mask(y)
        for decoder in self.decoder_layers:
            y = decoder(y, x, seq_mask)

        return y


