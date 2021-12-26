import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

x_train_rnn_path = "../data/X_train_rnn.npy"
y_train_path = "../data/y_train.npy"
x_test_rnn_path = "../data/X_test_rnn.npy"
y_test_path = "../data/y_test.npy"

x_train_logistic = np.load(x_train_rnn_path, allow_pickle=True)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_rnn_path, allow_pickle=True)
y_test = np.load(y_test_path)

total_train = []
unique_item_ids = []
for i in x_train_logistic:
    general = i[0].tolist()
    chart_events = i[1]
    icu_stay = []
    item_ids = []
    for event in chart_events:
        item_id = event[1]
        value_num = event[2]
        general_value = general + [value_num]
        # create general info + value_num list for each itu_stay
        icu_stay.append(general_value)
        # create value_nums list for each itu_stay
        # add item_ids without duplicates - used for word embeddings
        if item_id not in unique_item_ids:
            unique_item_ids.append(item_id)
    total_train.append(icu_stay)

# sort unique item ids
unique_item_ids.sort()


# gru with word embeddings for item_id
class GRUNet(nn.Module):
    def __init__(self, input_dim, word_emb_len, embed_dim, hidden_dim, output_dim, n_layers, drop_prob=0.1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(word_emb_len, embed_dim)

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, item_x, h):
        # x shape: (batch_size, 100, general_dim)
        # shape: (batch_size, 100, emb_dim)
        embedded = self.embedding(item_x)
        x = torch.cat((x, embedded), 2)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:-1]))
        return out, h





