import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time

import rnn_options

op = rnn_options.Options()

x_train_rnn_path = "../data/X_train_rnn.npy"
y_train_path = "../data/y_train.npy"
x_test_rnn_path = "../data/X_test_rnn.npy"
y_test_path = "../data/y_test.npy"

x_train_logistic = np.load(x_train_rnn_path, allow_pickle=True)
y_train = np.load(y_train_path)
x_test_logistic = np.load(x_test_rnn_path, allow_pickle=True)
y_test = np.load(y_test_path)

# set device type to gpu / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # embedded shape: (batch_size, 100, emb_dim)
        embedded = self.embedding(item_x)
        x = torch.cat((x, embedded), 2)
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:-1]))
        return out, h


def train(train_loader, learn_rate=op.learn_rate, hidden_dim=op.hidden_dim, epochs=op.epochs):
    input_dim = next(iter(train_loader))[0][2]
    output_dim = 1
    n_layers = 2
    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # define loss function, optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    for epoch in range(epochs):
        start_time = time.clock()
        avg_loss = 0
        for x, label, index in train_loader:
            # item_x = item_id_loader[index]
            model.zero_grad()
            out, h = model(x.to(device).float())
            loss = criterion(out, label.to(device.float()))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        end_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}, Time: {} seconds".format(epoch+1, epochs, avg_loss/len(train_loader), \
                                                                          end_time-start_time))




