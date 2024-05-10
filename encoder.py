import torch
from torch import nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embedding_dim, train_CNN= False):
        super(EncoderCNN, self).__init__()

        self.train_CNN= train_CNN
        self.inception = models.inception_v3(pretrained= True, aux_logits= False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p= 0.5)
        self.setup_model()

    def forward(self, images):
        # images: [batch_size, c, w, h]
        features = self.inception(images)
        return self.dropout(self.relu(features))

    def setup_model(self):
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True

            else:
                param.requires_grad = self.train_CNN