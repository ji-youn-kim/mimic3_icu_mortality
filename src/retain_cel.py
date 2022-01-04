import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if op.model_type == "retain_general":
            return self.los[index], self.admit[index], self.insurance[index], self.lang[index], self.religion[index],\
                    self.marital[index], self.ethnicity[index], self.diagnosis[index], self.gender[index], \
                    self.item_id[index], self.value_num[index], self.before_pad_len[index], self.labels[index]

        elif op.model_type == "retain_only_chart_events":
            return self.item_id[index], self.value_num[index], self.before_pad_len[index], self.labels[index]


def general_collate_fn(batch):
    batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital, batch_ethnicity, \
        batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_before_pad_len, batch_labels = zip(*batch)

    # sort icu stay data by chart events count descending order
    zipped = zip(batch_before_pad_len, batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital,
                 batch_ethnicity, batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_labels)
    zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
    batch_before_pad_len, batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital,\
        batch_ethnicity, batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_labels = zip(*zipped)

    batch_general = [batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital,
                     batch_ethnicity, batch_diagnosis, batch_gender]

    # convert each list to np array
    batch_general = np.array(batch_general)
    batch_item_id = np.array([np.array([b_id]) for b_id in batch_item_id]).astype('int')
    batch_value_num = np.array([np.array([b_vn]) for b_vn in batch_value_num]).astype('float')

    batch_item_id = np.squeeze(batch_item_id, axis=1)
    batch_value_num = np.squeeze(batch_value_num, axis=1)

    return torch.tensor(batch_general), torch.tensor(batch_item_id), torch.tensor(batch_value_num), \
        torch.tensor(batch_labels), batch_before_pad_len


def chart_events_collate_fn(batch):
    batch_item_id, batch_value_num, batch_before_pad_len, batch_labels = zip(*batch)

    # sort icu stay data by chart events count descending order
    zipped = zip(batch_before_pad_len, batch_item_id, batch_value_num, batch_labels)
    zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
    batch_before_pad_len, batch_item_id, batch_value_num, batch_labels = zip(*zipped)

    # convert each list to np array
    batch_item_id = np.array([np.array([b_id]) for b_id in batch_item_id]).astype('int')
    batch_value_num = np.array([np.array([b_vn]) for b_vn in batch_value_num]).astype('float')

    batch_item_id = np.squeeze(batch_item_id, axis=1)
    batch_value_num = np.squeeze(batch_value_num, axis=1)

    return torch.tensor(batch_item_id), torch.tensor(batch_value_num), torch.tensor(batch_labels), batch_before_pad_len


class RETAIN(nn.Module):
    def __init__(self, dict_len_list, dim_input=op.dim_input, dim_emb=op.ce_dim_emb, dropout_input=op.dropout_input,
                 dropout_emb=op.dropout_emb, dim_alpha=op.dim_alpha, dim_beta=op.dim_beta,
                 dropout_context=op.dropout_context, dim_output=op.dim_output, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first

        # embedding for categorical features
        if op.model_type == "retain_general":
            self.admit_embedding = nn.Embedding(dict_len_list[0], dim_input)
            self.insurance_embedding = nn.Embedding(dict_len_list[1], dim_input)
            self.lang_embedding = nn.Embedding(dict_len_list[2], dim_input)
            self.religion_embedding = nn.Embedding(dict_len_list[3], dim_input)
            self.marital_embedding = nn.Embedding(dict_len_list[4], dim_input)
            self.ethnicity_embedding = nn.Embedding(dict_len_list[5], dim_input)
            self.diagnosis_embedding = nn.Embedding(dict_len_list[6], dim_input)
            self.gender_embedding = nn.Embedding(dict_len_list[7], dim_input)
            self.item_id_embedding = nn.Embedding(dict_len_list[8], dim_input)

            concat_dim_input = dim_input * 9 + 1 * 2

            # final embedding for concatenated feature embeddings
            self.embedding = nn.Sequential(
                nn.Dropout(p=dropout_input),
                nn.Linear(concat_dim_input, dim_emb).double(),
                nn.Dropout(p=dropout_emb)
            )

        if op.model_type == "retain_only_chart_events":
            self.item_id_embedding = nn.Embedding(dict_len_list[0], dim_input)
            self.embedding_dropout = nn.Dropout(p=dropout_input)

            concat_dim_input = dim_input + 1

            self.embedding = nn.Sequential(
                nn.Linear(concat_dim_input, dim_emb).double(),
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

    def forward(self, general, item, val, lengths):
        batch_size = item.size(dim=0)

        if op.model_type == "retain_general":
            admit_emb = self.admit_embedding(general[1].int())
            insurance_emb = self.insurance_embedding(general[2].int())
            lang_emb = self.lang_embedding(general[3].int())
            rel_emb = self.religion_embedding(general[4].int())
            marital_emb = self.marital_embedding(general[5].int())
            eth_emb = self.ethnicity_embedding(general[6].int())
            diag_emb = self.diagnosis_embedding(general[7].int())
            gender_emb = self.gender_embedding(general[8].int())
            item_emb = self.item_id_embedding(item.int())

            los = torch.unsqueeze(general[0], 1)
            # concatenate general features
            x_general = torch.cat((admit_emb, insurance_emb, lang_emb, rel_emb, marital_emb,
                                   eth_emb, diag_emb, gender_emb, los), 1)
            val = torch.unsqueeze(val, 2)
            # concatenate chart events
            x_chart_events = torch.cat((item_emb, val), 2)
            x_general = torch.unsqueeze(x_general, 1).expand(-1, 100, -1)
            # concatenate all features per icu stay
            # x shape: (batch_size, 100, concat_dim_input)
            x = torch.cat((x_general, x_chart_events), dim=2)

            # emb shape: (batch_size, 100, dim_emb)
            emb = self.embedding(x).float()

        if op.model_type == "retain_only_chart_events":
            item_emb = self.item_id_embedding(item.int())
            item_emb = self.embedding_dropout(item_emb)

            val = torch.unsqueeze(val, 2)
            # concatenate chart events
            # x shape: (batch_size, 100, dim_input + 1)
            x = torch.cat((item_emb, val), 2)

            # emb shape: (batch_size, 100, dim_emb)
            emb = self.embedding(x).float()

        # length shape: batch_size, type: list
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)
        # alpha_unpacked shape: (batch_size, 100, dim_alpha)
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
        # e shape: (batch_size, 100, 1)
        e = self.alpha_fc(alpha_unpacked)

        # make padding alpha values to zeros
        mask = torch.FloatTensor(
            [[[1.0] if i < lengths[idx] else [0.0] for i in range(100)] for idx in range(batch_size)]
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

        return logit, alpha, beta


def main():
    # define dataset, dataloader
    train_set = ChartEventSequenceWithLabelDataset(x_train_rnn, y_train)
    test_set = ChartEventSequenceWithLabelDataset(x_test_rnn, y_test)

    # define device type - gpu, cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if op.model_type == "retain_general":
        train_loader = DataLoader(dataset=train_set, batch_size=op.batch_size, shuffle=True,
                                  collate_fn=general_collate_fn)
        test_loader = DataLoader(dataset=test_set, batch_size=op.batch_size, shuffle=True,
                                 collate_fn=general_collate_fn)
        # define model and set to device
        model = RETAIN(dict_len_list=[admit_dict_len, insurance_dict_len, lang_dict_len, religion_dict_len,
                                      marital_dict_len, ethnicity_dict_len, diagnosis_dict_len,
                                      gender_dict_len, item_ids_dict_len])

    elif op.model_type == "retain_only_chart_events":
        train_loader = DataLoader(dataset=train_set, batch_size=op.batch_size, shuffle=True,
                                  collate_fn=chart_events_collate_fn)
        test_loader = DataLoader(dataset=test_set, batch_size=op.batch_size, shuffle=True,
                                 collate_fn=chart_events_collate_fn)
        # define model and set to device
        model = RETAIN(dict_len_list=[item_ids_dict_len])

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
            if op.model_type == "retain_general":
                batch_general, batch_item_id, batch_value_num, batch_labels, batch_before_pad_len = batch

            if op.model_type == "retain_only_chart_events":
                batch_item_id, batch_value_num, batch_labels, batch_before_pad_len = batch
                batch_general = torch.tensor([])

            batch_general = batch_general.to(device)
            batch_item_id = batch_item_id.to(device)
            batch_value_num = batch_value_num.to(device)
            batch_labels = batch_labels.to(device)

            # output shape: (batch_size, dim_output)
            output, alpha, beta = model(batch_general, batch_item_id, batch_value_num, batch_before_pad_len)

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
            if op.model_type == "retain_general":
                batch_general, batch_item_id, batch_value_num, batch_labels, batch_before_pad_len = batch

            if op.model_type == "retain_only_chart_events":
                batch_item_id, batch_value_num, batch_labels, batch_before_pad_len = batch
                batch_general = torch.tensor([])

            batch_general = batch_general.to(device)
            batch_item_id = batch_item_id.to(device)
            batch_value_num = batch_value_num.to(device)
            batch_labels = batch_labels.to(device)

            # output shape: (batch_size, dim_output)
            output, alpha, beta = model(batch_general, batch_item_id, batch_value_num, batch_before_pad_len)

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
