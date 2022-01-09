import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix

import rnn_options
from utils.rnn_word2id import *

op = rnn_options.Options()

x_train_rnn_path = "../data/X_train_rnn_indv.npy"
y_train_path = "../data/y_train_rem_nochev.npy"
x_test_rnn_path = "../data/X_test_rnn_indv.npy"
y_test_path = "../data/y_test_rem_nochev.npy"
trained_model_path = "../data/trained_model_path.pt"

x_train_rnn = np.load(x_train_rnn_path, allow_pickle=True)
y_train = np.load(y_train_path)
x_test_rnn = np.load(x_test_rnn_path, allow_pickle=True)
y_test = np.load(y_test_path)

# set device type to gpu / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChartEventSequenceWithLabelDataset(Dataset):
    def __init__(self, inputs, labels, reverse=True):

        labels = np.squeeze(labels, axis=1)

        if len(inputs[0]) != len(labels):
            raise ValueError("Inputs and Labels have different lengths")

        self.num_features = len(inputs)

        # if reverse = True, reverse item_ids, value_nums for RETAIN
        # does not reverse padded elements(0) - elements only before padding are reversed
        if reverse:
            item_id_total = inputs[self.num_features - 3]
            value_num_total = inputs[self.num_features - 2]
            before_pad_len_total = inputs[self.num_features - 1]
            for index, item_list in enumerate(item_id_total):
                before_pad_len_cur = before_pad_len_total[index]
                inputs[self.num_features - 3][index][:before_pad_len_cur] = item_list[:before_pad_len_cur][::-1]
            for index, value_num_list in enumerate(value_num_total):
                before_pad_len_cur = before_pad_len_total[index]
                inputs[self.num_features - 2][index][:before_pad_len_cur] = value_num_list[:before_pad_len_cur][::-1]

        self.los, self.admit, self.insurance, self.lang, self.religion, self.marital, self.ethnicity, self.diagnosis,\
            self.gender, self.item_id, self.value_num, self.before_pad_len = inputs
        self.labels = labels

        self.seq = []
        # create a coo_matrix for each icu_stay
        for index_i, icu_item_list in enumerate(self.item_id):
            row = []
            col = []
            val = []
            # coo_matrix shape: (time_step, features)
            # retain_general : (100, 1 + admit_count + insurance_count + lang_count + religion_count +marital_count
            # + ethnicity_count + diagnosis_count + gender_count + item_id_count + 1)
            # chart_events_only: (100, item_id_count + 1)
            for idx, item in enumerate(icu_item_list):
                col_len = 0

                # one-hot encoding in row for item_id
                row.append(idx)
                col.append(col_len + item)
                val.append(1.0)
                col_len += item_ids_dict_len

                # value_num value in row for value_num
                row.append(idx)
                col.append(col_len)
                val.append(self.value_num[index_i][idx])
                col_len += 1

                if op.model_type == "retain_general":

                    # los value in row for los
                    row.append(idx)
                    col.append(col_len)
                    val.append(self.los[index_i])
                    col_len += 1

                    # one-hot encoding in row for admit, insurance, lang, religion, marital, eth, diag, gender
                    row.append(idx)
                    col.append(col_len + self.admit[index_i])
                    val.append(1.0)
                    col_len += admit_dict_len

                    row.append(idx)
                    col.append(col_len + self.insurance[index_i])
                    val.append(1.0)
                    col_len += insurance_dict_len

                    row.append(idx)
                    col.append(col_len + self.lang[index_i])
                    val.append(1.0)
                    col_len += lang_dict_len

                    row.append(idx)
                    col.append(col_len + self.religion[index_i])
                    val.append(1.0)
                    col_len += religion_dict_len

                    row.append(idx)
                    col.append(col_len + self.marital[index_i])
                    val.append(1.0)
                    col_len += marital_dict_len

                    row.append(idx)
                    col.append(col_len + self.ethnicity[index_i])
                    val.append(1.0)
                    col_len += ethnicity_dict_len

                    row.append(idx)
                    col.append(col_len + self.diagnosis[index_i])
                    val.append(1.0)
                    col_len += diagnosis_dict_len

                    row.append(idx)
                    col.append(col_len + self.gender[index_i])
                    val.append(1.0)
                    col_len += gender_dict_len

            self.seq.append(coo_matrix((np.array(val, dtype='float'), (np.array(row), np.array(col))),
                            shape=(len(icu_item_list), col_len)).toarray())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seq[index], self.before_pad_len[index], self.labels[index]


def collate_fn(batch):
    # sort batch [(seq1, before_pad_len, label), (seq2, before_pad_len, label), ...] by before_pad_len descending order
    batch = sorted(batch, key=lambda x: x[1], reverse=True)

    # batch_seq: [seq1_array, seq1_array, ...], type: list of arrays
    # batch_before_pad_len: [5, 2, ...], type: list of numbers
    batch_seq, batch_before_pad_len, batch_labels = zip(*batch)

    return torch.tensor(batch_seq), torch.tensor(batch_labels), batch_before_pad_len


class RETAIN(nn.Module):
    def __init__(self, dim_input, dim_emb=op.dim_emb, dropout_input=op.dropout_input,
                 dropout_emb=op.dropout_emb, dim_alpha=op.dim_alpha, dim_beta=op.dim_beta,
                 dropout_context=op.dropout_context, dim_output=op.dim_output, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first

        self.embedding = nn.Sequential(
            nn.Linear(dim_input, dim_emb).double(),
            nn.Dropout(p=dropout_emb)
        )

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)
        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)
        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context),
            nn.Linear(in_features=dim_emb, out_features=dim_output)
        )

    @staticmethod
    def masked_softmax(batch_tensor, mask):
        exp = torch.exp(batch_tensor)
        masked_exp = torch.mul(exp, mask)
        sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
        return masked_exp / sum_masked_exp

    def forward(self, seq, lengths):
        batch_size = seq.size(dim=0)
        max_len = seq.size(dim=1)

        emb = self.embedding(seq).float()

        # length shape: batch_size, type: list
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)
        # alpha_unpacked shape: (batch_size, 100, dim_alpha)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
        # e shape: (batch_size, 100, 1)
        e = self.alpha_fc(alpha_unpacked)

        # make padding alpha values to zeros
        mask = torch.FloatTensor(
            [[[1.0] if i < lengths[idx] else [0.0] for i in range(max_len)] for idx in range(batch_size)]
        )

        if next(self.parameters()).is_cuda:
            mask = mask.cuda()

        # alpha size: (batch_size, 100, 1)
        alpha = self.masked_softmax(e, mask)

        # h shape: (batch_size, 100, dim_beta)
        h, _ = self.rnn_beta(packed_input)
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)
        # beta shape: (batch_size, 100, dim_emb)
        beta = torch.tanh(self.beta_fc(beta_unpacked))

        # alpha size: (batch_size, 100, 1)
        # beta shape: (batch_size, 100, dim_emb)
        # emb shape: (batch_size, 100, dim_emb)
        # context shape: (batch_size, dim_emb)
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # logit shape: (batch_size, dim_output)
        logit = self.output(context)

        # w_emb size: (dim_emb, dim_input)
        w_emb = self.embedding[0].weight
        # w size: (dim_output, dim_emb)
        w = self.output[1].weight

        return logit, alpha, beta, w_emb, w


def main():
    # define dataset, dataloader
    train_set = ChartEventSequenceWithLabelDataset(x_train_rnn, y_train)
    test_set = ChartEventSequenceWithLabelDataset(x_test_rnn, y_test)

    # define device type - gpu, cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(dataset=train_set, batch_size=op.batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=op.batch_size, shuffle=True,
                             collate_fn=collate_fn, drop_last=True)

    if op.model_type == "retain_general":
        dim_input = 1 + admit_dict_len + insurance_dict_len + lang_dict_len + religion_dict_len + marital_dict_len \
                    + ethnicity_dict_len + diagnosis_dict_len + gender_dict_len + item_ids_dict_len + 1

    elif op.model_type == "retain_only_chart_events":
        dim_input = item_ids_dict_len + 1

    print("dim_input: ", dim_input)

    # define model and set to device
    model = RETAIN(dim_input=dim_input)
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=op.lr, weight_decay=op.weight_decay)

    model.train()

    for e_i in range(op.epochs):
        labels = []
        outputs = []
        total_loss = 0
        train_count = 0
        for b_i, batch in enumerate(tqdm(train_loader)):

            batch_seq, batch_labels, batch_before_pad_len, = batch

            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)

            # output shape: (batch_size, dim_output)
            # output, alpha, beta = model(batch_general, batch_item_id, batch_value_num, batch_before_pad_len)
            output, alpha, beta, _, _ = model(batch_seq, batch_before_pad_len)

            # batch labels shape: (batch_size, dim_output)
            loss = criterion(output, batch_labels.long())
            softmax = nn.Softmax(dim=1)

            labels.append(batch_labels)
            # output shape: (batch_size, 2)
            outputs.append(softmax(output))

            total_loss += loss.item()
            train_count += batch_labels.size(dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Total Train count: ", train_count)
        print("Epoch: ", e_i+1, " Average Loss: ", total_loss/train_count)

        train_auroc = roc_auc_score(torch.cat(labels, 0).cpu().detach().numpy(),
                                    torch.cat(outputs, 0).cpu().detach().numpy()[:, 1])
        train_auprc = average_precision_score(torch.cat(labels, 0).cpu().detach().numpy(),
                                              torch.cat(outputs, 0).cpu().detach().numpy()[:, 1])

        print("RETAIN TRAIN AUROC: ", train_auroc)
        print("RETAIN TRAIN AUPRC: ", train_auprc)

    torch.save(model.state_dict(), trained_model_path)

    test_labels = []
    test_outputs = []
    total_loss = 0
    test_count = 0

    model.eval()

    with torch.no_grad():
        for b_i, batch in enumerate(tqdm(test_loader)):
            batch_seq, batch_labels, batch_before_pad_len = batch

            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)

            # interpretability
            # alpha size: (batch_size, 100, 1)
            # beta size: (batch_size, 100, dim_emb)
            # w_emb size: (dim_emb, dim_input)
            # w size: (dim_output, dim_emb)
            # output shape: (batch_size, dim_output)
            output, alpha, beta, _, _ = model(batch_seq, batch_before_pad_len)

            # batch labels shape: (batch_size, dim_output)
            loss = criterion(output, batch_labels.long())
            softmax = nn.Softmax(dim=1)

            test_labels.append(batch_labels)
            test_outputs.append(softmax(output))

            total_loss += loss.item()
            test_count += batch_labels.size(dim=0)

        print("Total Test count: ", test_count)
        print("Average Loss: ", total_loss / test_count)

        test_auroc = roc_auc_score(torch.cat(test_labels, 0).cpu().detach().numpy(),
                                   torch.cat(test_outputs, 0).cpu().detach().numpy()[:, 1])
        test_auprc = average_precision_score(torch.cat(test_labels, 0).cpu().detach().numpy(),
                                             torch.cat(test_outputs, 0).cpu().detach().numpy()[:, 1])

        print("RETAIN TEST AUROC: ", test_auroc)
        print("RETAIN TEST AUPRC: ", test_auprc)


main()
