#!/usr/bin/python3 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataProcess import load_data
from torch.utils.data import DataLoader, TensorDataset

class ReviewSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.softMax = nn.LogSoftmax(dim=1)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, text, hidden):
        batch_size = text.size(0)
        embeds = self.embedding(text)
        lstm_out, hidden = self.lstm(embeds, hidden) 
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        #out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        
        #out = out[:, -1, :]
        #out = self.fc(out.view(-1, out.size(2)))
        softmax_out = self.softMax(out)
        softmax_out = softmax_out.view(batch_size, -1)
        #softmax_out = softmax_out[:, -1]

        return softmax_out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True)
    correct = max_preds.squeeze(1).eq(y)

    return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, data_loader, validate_loader, optimizer, criterion, batch_size, validate_counter = 100):
    epoch_loss = 0
    epoch_acc = 0
    clip = 5
    counter = 0
    
    model.train()

    hidden = model.init_hidden(batch_size)

    for reviews, labels in data_loader:
        counter += 1
        optimizer.zero_grad()
        hidden = tuple([each.data for each in hidden])    

        reviews = reviews.type(torch.LongTensor)
        predictions, hidden = model(reviews, hidden)
        loss = criterion(predictions.squeeze(), labels.long())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        acc = categorical_accuracy(predictions, labels)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        #print("Loss = " + str(loss.item()))
        #print("Accuracy = " + str(acc.item()))
        #print(epoch_loss)
        #print(epoch_acc)

        if counter % validate_counter == 0:
            validate_hidden = model.init_hidden(batch_size)        
            validate_loss = []
            validate_acc = []
            model.eval()

            for rev, lab in validate_loader:
                validate_hidden = tuple([each.data for each in validate_hidden])
                
                rev = rev.type(torch.LongTensor)
                preds, hid = model(rev, validate_hidden)
                val_loss = criterion(preds.squeeze(), lab.long())                
                val_acc = categorical_accuracy(preds, lab)

                validate_loss.append(val_loss.item())
                validate_acc.append(val_acc.item())

            model.train()

            print("Step: {}...".format(counter), "Step Loss: {:.6f}...".format(epoch_loss/counter), "Step Acc: {:.6f}...".format(epoch_acc/counter), 
                    "Validate mean Loss: {:.6f}".format(np.mean(validate_loss)), "Validate mean Acc: {:.6f}".format(np.mean(validate_acc)))


def main():
    data_split_ratio = 0.9
    batch_size = 256

    output_dim = 5
    embedding_dim = 300
    hidden_dim = 256
    n_layers = 2
    learning_rate = 0.001
    epoch = 1

    vocabulary, data_reviews, data_label = load_data(pad=True, plot=False) 

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
    dev_test_label = tmp_data[int(len(tmp_label) * 0.5):]

    print("Encoded Reviews length = %d; Label Data length = %d, Vocabulary length = %d"% (rev_len, label_len, vocab_len))
    print("Train Data length = %d; Train Label length = %d"%(len(train_data), len(train_label)))
    print("Validata Data length = %d; Validata Label length = %d"%(len(validate_data), len(validate_label)))
    print("Dev/Test Data length = %d; Dev/Test Label length = %d"%(len(dev_test_data), len(dev_test_label)))

    
    train_tensor = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label))
    validate_tensor = TensorDataset(torch.from_numpy(validate_data), torch.from_numpy(validate_label))
    dev_test_tensor = TensorDataset(torch.from_numpy(dev_test_data), torch.from_numpy(dev_test_label))

    train_loader = DataLoader(train_tensor, shuffle=False, batch_size=batch_size)
    validate_loader = DataLoader(validate_tensor, shuffle=False, batch_size=batch_size)
    dev_test_loader = DataLoader(dev_test_tensor, shuffle=False, batch_size=batch_size)

    train_iter = iter(train_loader)
    x, y = train_iter.next()

    #print(x.size())
    #print(y.size())


    model = ReviewSentiment(vocab_len, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0)

    print("Number of parameters in the model = %d." %(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for ep in range(epoch):
        train(model, train_loader, validate_loader, optimizer, criterion, batch_size, validate_counter=100)

if __name__ == '__main__':
    main()


