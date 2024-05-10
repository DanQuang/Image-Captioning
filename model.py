import torch
from torch import nn
from encoder import EncoderCNN
from decoder import DecoderRNN


class ImageCaptionModel(nn.Module):
    def __init__(self, config, vocab_size, padding_idx):
        super(ImageCaptionModel, self).__init__()
        self.embedding_dim = config["text_embedding"]["embedding_dim"]
        self.hidden_size = config["model"]["hidden_size"]
        self.num_layers = config["model"]["num_layers"]
        self.encoder = EncoderCNN(self.embedding_dim)
        self.decoder = DecoderRNN(self.embedding_dim, self.hidden_size, self.num_layers, vocab_size, padding_idx)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs