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
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.softMax = nn.Softmax(dim=1)

    def forward(self, text):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = text.size(0)
                
        #print("batch shape {}.".format(batch_size))

        h0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.n_layers*2, batch_size, self.hidden_dim).requires_grad_().to(device)

        #print("h0 shape {}.".format(h0.shape), "c0 shape {}.".format(c0.shape))

        embeds = self.embedding(text)
        #print("embeds shape {}.".format(embeds.shape))


        lstm_out, (h, c) = self.lstm(embeds, (h0.detach(), c0.detach()))
        #print("lstm_out shape {}.".format(lstm_out.shape), "h shape {}.".format(h.shape), "c shape {}.".format(c.shape))
 

        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        #hidden = self.dropout(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1))
        #print("hidden shape {}.".format(hidden.shape))
        out = self.dropout((torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1)))
        #print("dropout shape {}.".format(out.shape))
        out = self.fc(out)
        #print("fc out shape {}.".format(out.shape))
        #out = out.view(batch_size, -1)
        #print("fc out shape {}.".format(out.shape))
        
        #out = self.fc(out.view(-1, out.size(2)))
        #softmax_out = self.softMax(out)
        #print("softmax_out shape {}.".format(softmax_out.shape))
        
        
        #softmax_out = softmax_out.view(batch_size, -1)
        #softmax_out = softmax_out[:, -1]
        #print("softmax_out shape {}.".format(softmax_out.shape))

        return out


def categorical_accuracy(preds, y):
    #max_preds = preds.argmax(dim = 1, keepdim = True)
    #correct = max_preds.squeeze(1).eq(y)    

    _, max_preds = torch.max(preds.data, 1)

    if torch.cuda.is_available():
        correct = (max_preds.cpu() == y.cpu()).sum()
    else:
        correct = (max_preds == y).sum()

    return correct
    #return correct.sum() / torch.FloatTensor([y.shape[0]])

def train(model, device, data_loader, validate_loader, optimizer, criterion, batch_size, validate_counter = 100):
    epoch_loss = 0
    epoch_acc = 0
    clip = 5
    counter = 0
    total = 0    

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
        
        total += labels.size(0)
        epoch_acc += categorical_accuracy(predictions, labels)
                
        if counter % validate_counter == 0:
            #validate_loss.append(val_loss.item())
            accuracy = 100*epoch_acc/total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(counter, loss.item(), accuracy))

    accuracy = 100*epoch_acc/total
    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(counter, loss.item(), accuracy))

'''
        if counter % validate_counter == 0:
            validate_loss = []
            validate_acc = 0
            total = 0

            model.eval()

            for rev, lab in validate_loader:
                rev = rev.to(device)
                lab = lab.to(device)
             
                rev = rev.type(torch.LongTensor)
                lab = lab.type(torch.LongTensor)

                preds = model(rev)
                val_loss = criterion(preds, lab)                

                total += lab.size(0)
                validate_acc += categorical_accuracy(preds, lab)
                validate_loss.append(val_loss.item())

            model.train()

            accuracy = 100*validate_acc/total
            print('Iteration: {}. Loss: {}. Validate Set mean Loss {}. Validate Set Accuracy: {}'.format(counter, loss.item(), np.mean(validate_loss), accuracy))
'''

def main():
    data_split_ratio = 0.92
    batch_size = 512

    output_dim = 5
    embedding_dim = 300
    hidden_dim = 256
    n_layers = 2
    learning_rate = 0.01
    epoch = 1
    dropout = 0.5

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

    train_loader = DataLoader(train_tensor, shuffle=True, batch_size=batch_size)
    validate_loader = DataLoader(validate_tensor, shuffle=True, batch_size=batch_size)
    dev_test_loader = DataLoader(dev_test_tensor, shuffle=True, batch_size=batch_size)

    train_iter = iter(train_loader)
    x, y = train_iter.next()

    print("train_iter - batch size for text data is " + str(x.size()))
    print("train_iter - batch size for label data is " + str(y.size()))

    if torch.cuda.is_available():
        print("CUDA is AVAILABLE!!")

    model = ReviewSentiment(vocab_len, embedding_dim, hidden_dim, output_dim, n_layers, dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Number of parameters in the model = %d." %(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    
    for ep in range(epoch):
        print("Epoch {}.".format(ep))
        train(model, device, train_loader, validate_loader, optimizer, criterion, batch_size, validate_counter=25)

if __name__ == '__main__':
    main()


