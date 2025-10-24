import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralRecommender(nn.Module):
    def __init__(self, input_size, embedding_dim=32):
        super(NeuralRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)