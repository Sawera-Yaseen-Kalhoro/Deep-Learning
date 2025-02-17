import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import *
from colors import *
import random

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size // 2)  # Output vector half of original size (dimensionlaity reduction)
        self.w2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, h):
        a = self.w2(torch.tanh(self.W1(h))) # Non-linnear transformation of the hidden states
        alpha = torch.nn.functional.softmax(a, dim=0)  # Compute attention weights
        out_attn = torch.sum(alpha * h, dim=0)  # Weighted sum of hidden states
        return out_attn


class Recurrent(nn.Module):
    def __init__(self, type='GRU', hidden_size=150, num_layers=2, dropout=0.5, bidirectional=False, use_attention=False):
        super(Recurrent, self).__init__()
        self.type = type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        if self.type == 'GRU':
            self.rnn = nn.GRU(input_size=300,
                              hidden_size=self.hidden_size, 
                              batch_first=False, 
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              bidirectional=self.bidirectional
                              )
        elif self.type == 'LSTM':
            self.rnn = nn.LSTM(input_size=300,
                              hidden_size=self.hidden_size, 
                              batch_first=False, 
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              bidirectional=self.bidirectional
                              )
        else:
            self.rnn = nn.RNN(input_size=300,
                              hidden_size=self.hidden_size, 
                              batch_first=False, 
                              num_layers=self.num_layers,
                              dropout=self.dropout,
                              bidirectional=self.bidirectional
                              )
        
        if self.use_attention:
            self.attention = BahdanauAttention(self.hidden_size)
        
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        if self.type == 'LSTM':
            out, (x, _) = self.rnn(x)   # For LSTMs x consists of (hidden state, cell state)
        else:
            out, x = self.rnn(x)        # For GRU/RNN only hidden state is needed

        if self.use_attention:
            x = self.attention(out)     # Apply attention over all time steps
        else:
            x = x[-1]                   # Take the last hidden state

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, dataloader, epochs, optimizer, loss_fn, embeddings, validation_dataloader, device):
    print('Starting Training')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (words, labels, lengths) in enumerate(dataloader):
            labels = labels.to(device)
            words = words.to(device)
            optimizer.zero_grad()
            words = words.type(torch.LongTensor)
            words = embeddings(words).to(device)
            # current shape of words is (batch_size, max_length, embedding_size)
            # convert to time first format
            words = words.transpose(1, 0)
            output = model.forward(words).squeeze()
            loss = loss_fn(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
        accuracy, CM, _, _ , f1= evaluate(model, validation_dataloader, loss_fn, embeddings, device)
        print(Colors.RED+f'Epoch {epoch}: Valid accuracy: {accuracy.item()}' + Colors.RESET)
    print('Finished Training')
    return accuracy, f1
    

def evaluate(model, dataloader, loss_fn, embeddings, device):
    model.eval()
    confusion_matrix = torch.zeros(2, 2)
    # determine the confusion matrix for the data_loader
    with torch.no_grad():
        for i, (words, labels, lengths) in enumerate(dataloader):
            labels = labels.to(device)
            words = words.to(device)
            words = words.type(torch.LongTensor)
            words = embeddings(words).to(device)
            words = words.transpose(1, 0)
            output = model.forward(words).squeeze()
            # loss = loss_fn(output, labels)
            predictions = torch.round(torch.sigmoid(output))
            for i in range(len(predictions)):
                confusion_matrix[int(labels[i]), int(predictions[i])] += 1
    
    # determine the overall accuracy using the confusion matrix
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()

    # determine the precision and recall for each class
    precision = confusion_matrix[0, 0] / confusion_matrix[0, :].sum()
    recall = confusion_matrix[0, 0] / confusion_matrix[:, 0].sum()
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, confusion_matrix, precision, recall, f1

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)
    seed=7052020
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = 10
    shuffle = True
    train_dataset = NLPDataset('data/sst_train_raw.csv', train=True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fcn)
    
    text_vocab = Vocab(train_dataset.text_frequencies, max_size=-1, min_freq=0)
    label_vocab = Vocab(train_dataset.label_frequencies, max_size=-1, min_freq=0)
    
    test_dataset = NLPDataset('data/sst_test_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                shuffle=shuffle, collate_fn=pad_collate_fcn)
    
    validation_dataset = NLPDataset('data/sst_valid_raw.csv', train=False, text_vocab=text_vocab, label_vocab=label_vocab)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=len(validation_dataset), 
                                    collate_fn=pad_collate_fcn)

    word_rep = load_embeddings('data/sst_glove_6b_300d.txt')
    word_embeddings = gen_embeddings(text_vocab.stoi, word_rep)

    # pdb.set_trace()
    loss_fn = nn.BCEWithLogitsLoss()

    cell_type = ['LSTM','GRU', 'RNN']
    hidden_size = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    num_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    dropout = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]
    bidirectional = [True, False]
    results = []

    for type in cell_type:
        for i in range(2):
            hidden = 150
            layers = 2
            drop = 0.0
            attn = random.choice([False, False])  # Include attention or not
            print(Colors.BLUE+f'Cell type: {type}, Hidden size: {hidden}, Num layers: {layers}, Dropout: {drop}, Bidirectional: {False}, Attention: {attn}'+Colors.RESET)
            model = Recurrent(type=type, hidden_size=hidden, num_layers=layers, dropout=drop, bidirectional=False, use_attention=attn)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            accuracy, f1 = train(model, train_dataloader, 5, optimizer, loss_fn, word_embeddings, validation_dataloader, device)
            # round the accuracy to 2 decimal places
            accuracy = accuracy.item() * 100
            accuracy = round(accuracy, 2)
            # save accuracy for each model in a file
            results.append((type, hidden, layers, drop, False, accuracy, f1.item(), attn))
        
    # save the results list to a file
    with open('results/noAttentionresults.txt', 'w') as f:
        for item in results:
            f.write(f'Cell type: {item[0]}, Hidden size: {item[1]}, Num layers: {item[2]}, Dropout: {item[3]}, Bidirectional: {item[4]}, Accuracy: {item[5]}, F1 score: {item[6]}\n')

