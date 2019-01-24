from torch import nn

from ..aggregator.attention import MultiHeadAtte
from ..other_modules import LayerNormalization


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

        # def forward(self, input, seq_mask):
        def forward(self, input):
            attention = self.atte(input, input, input)
            norm_atte = self.norm1(attention + input)
            output = self.ffn(norm_atte)
            return self.norm2(output + norm_atte)

    def __init__(self, num_layers, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.Sequential(*[self.SubLayer(**kwargs) for _ in range(num_layers)])

    # def forward(self, x, seq_mask=None):
    def forward(self, x):
        # return self.layers(x, seq_mask)
        return self.layers(x)
