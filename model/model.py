#!/usr/bin/python3.7 
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataProcess import load_data
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel

class ReviewSentimentLstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, n_direction, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_direction = n_direction
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.bidirectional = self.n_direction == 2
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        self.fc = nn.Linear(hidden_dim * self.n_direction, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.softMax = nn.Softmax(dim=1)

    def forward(self, text):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = text.size(0)
                
        #print("batch shape {}.".format(batch_size))

        h0 = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.n_layers * self.n_direction, batch_size, self.hidden_dim).requires_grad_().to(device)

        #print("h0 shape {}.".format(h0.shape), "c0 shape {}.".format(c0.shape))

        embeds = self.embedding(text)
        #print("embeds shape {}.".format(embeds.shape))


        lstm_out, (h, c) = self.lstm(embeds, (h0.detach(), c0.detach()))
        #print("lstm_out shape {}.".format(lstm_out.shape), "h shape {}.".format(h.shape), "c shape {}.".format(c.shape))
  
        out = self.dropout(lstm_out)

        #out = self.dropout((torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)))
        #print("dropout shape {}.".format(out.shape))
        out = self.fc(out)
        #print("fc out shape {}.".format(out.shape))
        out = out[:, -1, :]
        #print("fc out shape {}.".format(out.shape))
        #out = out.view(batch_size, -1)
        #softmax_out = self.softMax(out)
        #print("softmax_out shape {}.".format(softmax_out.shape))
        
        
        #softmax_out = softmax_out.view(batch_size, -1)

        return out

class ReviewSentimentBert(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, n_direction, dropout):
        super().__init__()

        self.bert = bert
        self.hidden_dim  = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_direction = n_direction
        self.dropout = dropout

        self.bidirectional = self.n_direction == 2 
        self.embedding_dim = self.bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(self.embedding_dim, self.hidden_dim, num_layers = self.n_layers, bidirectional = self.bidirectional, batch_first = True, dropout=self.dropout)

        self.out = nn.Linear(self.hidden_dim * self.n_direction, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]

        _, hidden = self.rnn(embedded)

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        output = self.out(hidden)

        return output


def categorical_accuracy(preds, y):
    #max_preds = preds.argmax(dim = 1, keepdim = True)
    #correct = max_preds.squeeze(1).eq(y)    

    _, max_preds = torch.max(preds.data, 1)

    if torch.cuda.is_available():
        correct = (max_preds.cpu() == y.cpu()).sum()
    else:
        correct = (max_preds == y).sum()

    return correct

def test(model, device, test_loader, criterion):
    test_loss = []
    test_acc = 0

    model.eval()
    for rev, lab in test_loader: 

        rev = rev.type(torch.LongTensor)
        lab = lab.type(torch.LongTensor)

        rev = rev.to(device)
        lab = lab.to(device)
                
        preds = model(rev)
        loss = criterion(preds, lab)                

        test_acc += categorical_accuracy(preds, lab)
        test_loss.append(loss.item())

    accuracy = test_acc/len(test_loader.dataset)
    print('Test Set mean Loss {}. Test Set Accuracy: {}'.format(np.mean(test_loss), accuracy))



def train(model, device, data_loader, validate_loader, optimizer, criterion, validate_counter = 100):
    clip = 5
    counter = 0
    testloss = []
    test_acc = 0

    model.train()

    for reviews, labels in data_loader:
        counter += 1  

        optimizer.zero_grad()
        
        reviews = reviews.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor)

        reviews = reviews.to(device)
        labels = labels.to(device)
        
        predictions = model(reviews)

        loss = criterion(predictions, labels)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        
        #total += labels.size(0)
        #epoch_acc += categorical_accuracy(predictions, labels)
                
        if counter % validate_counter == 0:
            validate_loss = []
            validate_acc = 0
            #total = 0

            model.eval()

            for rev, lab in validate_loader: 
                rev = rev.type(torch.LongTensor)
                lab = lab.type(torch.LongTensor)

                rev = rev.to(device)
                lab = lab.to(device)
                
                preds = model(rev)
                val_loss = criterion(preds, lab)                

                #total += lab.size(0)
                validate_acc += categorical_accuracy(preds, lab)
                validate_loss.append(val_loss.item())

            model.train()

            accuracy = validate_acc/len(validate_loader.dataset)
            print('Iteration: {}. Loss: {}. Validate Set mean Loss {}. Validate Set Accuracy: {}'.format(counter, loss.item(), np.mean(validate_loss), accuracy))


    return model


def main(model_type='lstm'):
    data_split_ratio = 0.92
    batch_size = 256

    output_dim = 5
    embedding_dim = 400
    hidden_dim = 128
    n_layers = 4
    n_direction = 2
    learning_rate = 0.001
    epoch = 2
    dropout = 0.1

    vocabulary, data_reviews, data_label = load_data(hidden_dim, pad=True, plot=False) 

    rev_len = len(data_reviews)
    label_len = len(data_label)
    vocab_len = len(vocabulary)

    train_data = data_reviews[0:int(rev_len * data_split_ratio)]
    train_label = data_label[0:int(label_len * data_split_ratio)]

    tmp_data = data_reviews[int(rev_len * data_split_ratio):]
    tmp_label = data_label[int(label_len * data_split_ratio):]

    validate_data = tmp_data[0:int(len(tmp_data) * 0.5)]
    validate_label = tmp_label[0:int(len(tmp_label) * 0.5)]

    dev_test_data = tmp_data[int(len(tmp_data) * 0.5):]
    dev_test_label = tmp_label[int(len(tmp_label) * 0.5):]

    print("Encoded Reviews length = %d; Label Data length = %d, Vocabulary length = %d"% (rev_len, label_len, vocab_len))
    print("Train Data length = %d; Train Label length = %d"%(len(train_data), len(train_label)))
    print("Validata Data length = %d; Validata Label length = %d"%(len(validate_data), len(validate_label)))
    print("Dev/Test Data length = %d; Dev/Test Label length = %d"%(len(dev_test_data), len(dev_test_label)))

    
    train_tensor = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    validate_tensor = TensorDataset(torch.from_numpy(validate_data), torch.from_numpy(validate_label))
    dev_test_tensor = TensorDataset(torch.from_numpy(dev_test_data), torch.from_numpy(dev_test_label))

    train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
    validate_loader = DataLoader(validate_tensor, shuffle=True, batch_size=batch_size)
    dev_test_loader = DataLoader(dev_test_tensor, shuffle=False, batch_size=batch_size)

    print("Number of batches in Training Set = %d"%len(train_loader))
    print("Number of batches in Validation Set = %d"%len(validate_loader))
    print("Number of batches in Test Set = %d"%len(dev_test_loader))


    if torch.cuda.is_available():
        print("CUDA is AVAILABLE!!")

    print('Data Split Ratio: {}. Batch Size: {}. Number LSTM Units {}. Number LSTM Layers: {}. Dropout: {}. Direction: {}. Number Epoch: {}.'.format(data_split_ratio, batch_size, hidden_dim, n_layers, dropout, n_direction, epoch))
    print('Learning Rate: {}.'.format(learning_rate))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == 'lstm':
        print("Creating LSTM Model")
        model = ReviewSentimentLstm(vocab_len, embedding_dim, hidden_dim, output_dim, n_layers, n_direction, dropout)
        model.to(device)
    elif model_type == 'transformer':
        print("Creating BERT Model")
        device = torch.device("cpu")
        bert = BertModel.from_pretrained('bert-base-uncased')
        model = ReviewSentimentBert(bert, hidden_dim, output_dim, n_layers, n_direction, dropout)
        model.to(device)

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    print("Number of parameters in the model = %d." %(sum(p.numel() for p in model.parameters() if p.requires_grad)))
 
    print("Training the Model")
    for ep in range(epoch):
        t0 = time.time()
        print("Commencing Epoch {}.".format(ep))
        train(model, device, train_loader, validate_loader, optimizer, criterion, validate_counter=100)
        t1 = time.time()
        print("Epoch {}. took {}. seconds to complete".format(ep, (t1-t0)))


    print("Testing the Model")
    test(model, device, dev_test_loader, criterion)


if __name__ == '__main__':
    main('lstm')


