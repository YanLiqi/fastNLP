
from torch import nn
import torch
import torch.nn.functional as F

from fastNLP.modules.decoder.MLP import MLP
from fastNLP.models.base_model import BaseModel
from reproduction.chinese_word_segment.utils import seq_lens_to_mask

class CWSBiLSTMEncoder(BaseModel):
    def __init__(self, vocab_num, embed_dim=100, bigram_vocab_num=None, bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, bidirectional=True, embed_drop_p=None, num_layers=1):
        super().__init__()

        self.input_size = 0
        self.num_bigram_per_char = num_bigram_per_char
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embed_drop_p = embed_drop_p
        if self.bidirectional:
            self.hidden_size = hidden_size//2
            self.num_directions = 2
        else:
            self.hidden_size = hidden_size
            self.num_directions = 1

        if not bigram_vocab_num is None:
            assert not bigram_vocab_num is None, "Specify num_bigram_per_char."

        if vocab_num is not None:
            self.char_embedding = nn.Embedding(num_embeddings=vocab_num, embedding_dim=embed_dim)
            self.input_size += embed_dim

        if bigram_vocab_num is not None:
            self.bigram_embedding = nn.Embedding(num_embeddings=bigram_vocab_num, embedding_dim=bigram_embed_dim)
            self.input_size += self.num_bigram_per_char*bigram_embed_dim

        if not self.embed_drop_p is None:
            self.embedding_drop = nn.Dropout(p=self.embed_drop_p)

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=self.bidirectional,
                    batch_first=True, num_layers=self.num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'bias_hh' in name:
                nn.init.constant_(param, 0)
            elif 'bias_ih' in name:
                nn.init.constant_(param, 1)
            else:
                nn.init.xavier_uniform_(param)

    def init_embedding(self, embedding, embed_name):
        if embed_name == 'bigram':
            self.bigram_embedding.weight.data = torch.from_numpy(embedding)
        elif embed_name == 'char':
            self.char_embedding.weight.data = torch.from_numpy(embedding)


    def forward(self, chars, bigrams=None, seq_lens=None):

        batch_size, max_len = chars.size()

        x_tensor = self.char_embedding(chars)

        if not bigrams is None:
            bigram_tensor = self.bigram_embedding(bigrams).view(batch_size, max_len, -1)
            x_tensor = torch.cat([x_tensor, bigram_tensor], dim=2)

        sorted_lens, sorted_indices = torch.sort(seq_lens, descending=True)
        packed_x = nn.utils.rnn.pack_padded_sequence(x_tensor[sorted_indices], sorted_lens, batch_first=True)

        outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        _, desorted_indices = torch.sort(sorted_indices, descending=False)
        outputs = outputs[desorted_indices]

        return outputs


class CWSBiLSTMSegApp(BaseModel):
    def __init__(self, vocab_num, embed_dim=100, bigram_vocab_num=None, bigram_embed_dim=100, num_bigram_per_char=None,
                 hidden_size=200, bidirectional=True, embed_drop_p=None, num_layers=1, tag_size=2):
        super(CWSBiLSTMSegApp, self).__init__()

        self.tag_size = tag_size

        self.encoder_model = CWSBiLSTMEncoder(vocab_num, embed_dim, bigram_vocab_num, bigram_embed_dim, num_bigram_per_char,
                 hidden_size, bidirectional, embed_drop_p, num_layers)

        size_layer = [hidden_size, 100, tag_size]
        self.decoder_model = MLP(size_layer)


    def forward(self, batch_dict):
        device = self.parameters().__next__().device
        chars = batch_dict['indexed_chars_list'].to(device)
        if 'bigram' in batch_dict:
            bigrams = batch_dict['indexed_chars_list'].to(device)
        else:
            bigrams = None
        seq_lens = batch_dict['seq_lens'].to(device)

        feats = self.encoder_model(chars, bigrams, seq_lens)
        probs = self.decoder_model(feats)

        pred_dict = {}
        pred_dict['seq_lens'] = seq_lens
        pred_dict['pred_prob'] = probs

        return pred_dict

    def predict(self, batch_dict):
        pass


    def loss_fn(self, pred_dict, true_dict):
        seq_lens = pred_dict['seq_lens']
        masks = seq_lens_to_mask(seq_lens).float()

        pred_prob = pred_dict['pred_prob']
        true_y = true_dict['tags']

        # TODO 当前把loss写死了
        loss = F.cross_entropy(pred_prob.view(-1, self.tag_size),
                               true_y.view(-1), reduction='none')*masks.view(-1)/torch.sum(masks)


        return loss