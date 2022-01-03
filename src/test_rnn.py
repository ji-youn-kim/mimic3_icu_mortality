import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


import rnn_options
from utils.rnn_word2id import *

op = rnn_options.Options()

x_train_rnn_path = "../data/X_train_rnn_indv.npy"
y_train_path = "../data/y_train_rem_nochev.npy"
x_test_rnn_path = "../data/X_test_rnn_indv.npy"
y_test_path = "../data/y_test_rem_nochev.npy"

x_train_rnn = np.load(x_train_rnn_path, allow_pickle=True)
y_train = np.load(y_train_path)
x_test_rnn = np.load(x_test_rnn_path, allow_pickle=True)
y_test = np.load(y_test_path)

# set device type to gpu / cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("TRAIN=====================")
# print(x_train_rnn.shape)
# print(x_train_rnn[9][:2])
# print(x_train_rnn[10][:2])
# print(x_train_rnn[11][:2])
# print("TEST======================")
# print(x_test_rnn.shape)
# print(x_test_rnn[9][:2])
# print(x_test_rnn[10][:2])
# print(x_test_rnn[11][:2])
# print("Y_TRAIN====================")
# print(y_train.shape)
# print(y_train)
# print("Y_TEST====================")
# print(y_test.shape)
# print(y_test)


class ChartEventSequenceWithLabelDataset(Dataset):
    def __init__(self, inputs, labels, reverse=True):

        labels = np.squeeze(labels, axis=1)

        if len(inputs[0]) != len(labels):
            raise ValueError("Inputs and Labels have different lengths")

        self.num_features = len(inputs)

        # if reverse = True, reverse item_ids, value_nums for RETAIN
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
        return self.los[index], self.admit[index], self.insurance[index], self.lang[index], self.religion[index],\
                self.marital[index], self.ethnicity[index], self.diagnosis[index], self.gender[index], \
                self.item_id[index], self.value_num[index], self.before_pad_len[index], self.labels[index]


def visit_collate_fn(batch):
    batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital, batch_ethnicity, \
        batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_before_pad_len, batch_labels = zip(*batch)

    zipped = zip(batch_item_id, batch_value_num, batch_before_pad_len)
    zipped = sorted(zipped, key=lambda x: x[2], reverse=True)
    batch_item_id, batch_value_num, batch_before_pad_len = zip(*zipped)

    # print("np array itemid, valuenum print")
    # print(np.array([np.array([b_id]) for b_id in batch_item_id]))
    # print(np.array([np.array([b_vn]) for b_vn in batch_value_num]))

    batch_los = np.stack(np.array(batch_los), axis=0)
    batch_admit = np.stack(np.array(batch_admit), axis=0)
    batch_insurance = np.stack(np.array(batch_insurance), axis=0)
    batch_lang = np.stack(np.array(batch_lang), axis=0)
    batch_religion = np.stack(np.array(batch_religion), axis=0)
    batch_marital = np.stack(np.array(batch_marital), axis=0)
    batch_ethnicity = np.stack(np.array(batch_ethnicity), axis=0)
    batch_diagnosis = np.stack(np.array(batch_diagnosis), axis=0)
    batch_gender = np.stack(np.array(batch_gender), axis=0)
    batch_item_id = np.stack(np.array([np.array([b_id]) for b_id in batch_item_id]).astype('int'), axis=0)
    batch_value_num = np.stack(np.array([np.array([b_vn]) for b_vn in batch_value_num]).astype('float'), axis=0)

    batch_item_id = np.squeeze(batch_item_id, axis=1)
    batch_value_num = np.squeeze(batch_value_num, axis=1)
    # print("after stack")
    # print(batch_item_id)
    # print(type(batch_item_id))
    # print(type(batch_item_id[0]))
    # print(batch_item_id.shape)

    return torch.tensor(batch_los), torch.tensor(batch_admit), torch.tensor(batch_insurance), torch.tensor(batch_lang),\
        torch.tensor(batch_religion), torch.tensor(batch_marital), torch.tensor(batch_ethnicity),\
        torch.tensor(batch_diagnosis), torch.tensor(batch_gender), torch.tensor(batch_item_id), \
        torch.tensor(batch_value_num), torch.tensor(batch_labels), batch_before_pad_len


class RETAIN(nn.Module):
    def __init__(self, dim_input, dict_len_list, dim_emb=256, dropout_input=0.8, dropout_emb=0.9, dim_alpha=256,
                 dim_beta=256, dropout_context=0.6, dim_output=2, l2=0.0001, batch_first=True):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first

        # embedding for categorical features
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

    def forward(self, los, admit, insur, lang, rel, mar, eth, diag, gender, item, val, lengths):
        batch_size = item.size(dim=0)

        admit_emb = self.admit_embedding(admit)
        insurance_emb = self.insurance_embedding(insur)
        lang_emb = self.lang_embedding(lang)
        rel_emb = self.religion_embedding(rel)
        marital_emb = self.marital_embedding(mar)
        eth_emb = self.ethnicity_embedding(eth)
        diag_emb = self.diagnosis_embedding(diag)
        gender_emb = self.gender_embedding(gender)
        item_emb = self.item_id_embedding(item)

        los = torch.unsqueeze(los, 1)
        # concatenate general features
        x_general = torch.cat((admit_emb, insurance_emb, lang_emb, rel_emb, marital_emb, eth_emb, diag_emb, gender_emb, los), 1)
        print("x_general.shape: ", x_general.shape)
        val = torch.unsqueeze(val, 2)
        # concatenate chart events
        print("item_emb.shape: ", item_emb.shape)
        print("val.shape: ", val.shape)
        x_chart_events = torch.cat((item_emb, val), 2)
        x_general = torch.unsqueeze(x_general, 1).expand(-1, 100, -1)
        x = torch.cat((x_general, x_chart_events), dim=2)
        print("x shape: ", x.shape)
        # print("x[0][10]: ", x[0][10])
        emb = self.embedding(x).float()

        print(lengths)
        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        # g shape: (batch_size, 100, dim_emb)
        g, _ = self.rnn_alpha(packed_input)
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
        # shape: (batch_size, 100, dim_emb)
        beta = torch.tanh(self.beta_fc(beta_unpacked))

        # context shape: (batch_size, dim_emb)
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # logit shape: (batch_size, dim_output)
        logit = self.output(context)

        return logit, alpha, beta


def main():
    # define dataset, dataloader
    train_set = ChartEventSequenceWithLabelDataset(x_train_rnn, y_train)
    test_set = ChartEventSequenceWithLabelDataset(x_test_rnn, y_test)
    train_loader = DataLoader(dataset=train_set, batch_size=op.batch_size, shuffle=True, collate_fn=visit_collate_fn)
    test_loader = DataLoader(dataset=test_set, batch_size=op.batch_size, shuffle=True, collate_fn=visit_collate_fn)

    # define device type - gpu, cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model and set to device
    model = RETAIN(dim_input=256, dict_len_list=[admit_dict_len, insurance_dict_len, lang_dict_len, religion_dict_len,
                                                 marital_dict_len, ethnicity_dict_len, diagnosis_dict_len,
                                                 gender_dict_len, item_ids_dict_len])
    model = model.to(device)

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=op.lr)

    labels = []
    outputs = []
    total_loss = 0
    train_count = 0
    correct = 0

    for e_i in range(op.epochs):
        for b_i, batch in enumerate(tqdm(train_loader)):
            batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital, batch_ethnicity, \
                batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_labels, \
                batch_before_pad_len = batch

            batch_los = batch_los.to(device)
            batch_admit = batch_admit.to(device)
            batch_insurance = batch_insurance.to(device)
            batch_lang = batch_lang.to(device)
            batch_religion = batch_religion.to(device)
            batch_marital = batch_marital.to(device)
            batch_ethnicity = batch_ethnicity.to(device)
            batch_diagnosis = batch_diagnosis.to(device)
            batch_gender = batch_gender.to(device)
            batch_item_id = batch_item_id.to(device)
            batch_value_num = batch_value_num.to(device)
            batch_labels = batch_labels.to(device)

            # output shape: (batch_size, dim_output)
            output, alpha, beta = model(batch_los, batch_admit, batch_insurance, batch_lang, batch_religion,
                                        batch_marital, batch_ethnicity, batch_diagnosis, batch_gender, batch_item_id,
                                        batch_value_num, batch_before_pad_len)

            print("dtype: ", output.dtype, batch_labels.dtype)
            # batch labels shape: (batch_size, dim_output)
            print("batch_labels: ", batch_labels.tolist())
            print("batch_labels.size(): ", batch_labels.size())
            loss = criterion(output, batch_labels.long())
            labels.extend(batch_labels.tolist())
            _, pred_labels = torch.max(F.softmax(output).data, 1)
            outputs.extend(pred_labels.float().tolist())
            print("pred_labels.size: ", pred_labels.size())
            print("pred_labels: ", pred_labels.float().tolist())
            total_loss += loss.item()
            train_count += batch_labels.size(dim=0)
            correct += torch.eq(batch_labels, pred_labels).sum().item()
            print("correct elements: ", torch.eq(batch_labels, pred_labels).sum().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("Epoch: ", e_i, "Average Loss: ", total_loss / train_count)

        print("Total Train count: ", train_count)
        print("Epoch: ", e_i, "Average Loss: ", total_loss/train_count, "Accuracy: ", correct/train_count)

    train_auroc = roc_auc_score(labels, outputs)
    train_auprc = average_precision_score(labels, outputs)
    print("RETAIN TRAIN AUROC: ", train_auroc)
    print("RETAIN TRAIN AUPRC: ", train_auprc)

    test_labels = []
    test_outputs = []
    total_loss = 0
    test_count = 0
    correct = 0

    for b_i, batch in enumerate(tqdm(test_loader)):
        batch_los, batch_admit, batch_insurance, batch_lang, batch_religion, batch_marital, batch_ethnicity, \
            batch_diagnosis, batch_gender, batch_item_id, batch_value_num, batch_labels, \
            batch_before_pad_len = batch

        batch_los = batch_los.to(device)
        batch_admit = batch_admit.to(device)
        batch_insurance = batch_insurance.to(device)
        batch_lang = batch_lang.to(device)
        batch_religion = batch_religion.to(device)
        batch_marital = batch_marital.to(device)
        batch_ethnicity = batch_ethnicity.to(device)
        batch_diagnosis = batch_diagnosis.to(device)
        batch_gender = batch_gender.to(device)
        batch_item_id = batch_item_id.to(device)
        batch_value_num = batch_value_num.to(device)
        batch_labels = batch_labels.to(device)

        # output shape: (batch_size, dim_output)
        output, alpha, beta = model(batch_los, batch_admit, batch_insurance, batch_lang, batch_religion,
                                    batch_marital, batch_ethnicity, batch_diagnosis, batch_gender, batch_item_id,
                                    batch_value_num, batch_before_pad_len)

        # batch labels shape: (batch_size, dim_output)
        print("batch_labels: ", batch_labels.tolist())
        print("batch_labels.size(): ", batch_labels.size())
        loss = criterion(output, batch_labels.long())
        test_labels.extend(batch_labels.tolist())
        _, pred_labels = torch.max(F.softmax(output).data, 1)
        test_outputs.extend(pred_labels.float().tolist())
        print("pred_labels.size: ", pred_labels.size())
        print("pred_labels: ", pred_labels.float().tolist())
        total_loss += loss.item()
        test_count += batch_labels.size(dim=0)
        correct += torch.eq(batch_labels, pred_labels).sum().item()
        print("correct elements: ", torch.eq(batch_labels, pred_labels).sum().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Epoch: ", e_i, "Average Loss: ", total_loss / train_count)

    print("Total Test count: ", test_count)
    print("Epoch: ", e_i, "Average Loss: ", total_loss / test_count, "Accuracy: ", correct / test_count)

    test_auroc = roc_auc_score(test_labels, test_outputs)
    test_auprc = average_precision_score(test_labels, test_outputs)
    print("RETAIN TRAIN AUROC: ", test_auroc)
    print("RETAIN TRAIN AUPRC: ", test_auprc)


main()
