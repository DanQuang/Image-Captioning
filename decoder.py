import torch
from torch import nn


class DecoderRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers, output_size, padding_idx):
        super(DecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.embedding_dim, padding_idx= padding_idx)
        self.rnn = nn.LSTM(
            input_size= self.embedding_dim,
            hidden_size= self.hidden_size,
            num_layers= self.num_layers,
            batch_first= True
        )
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # features: encode from images
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim= 1)
        outputs, _ = self.rnn(embeddings)
        return self.linear(outputs)