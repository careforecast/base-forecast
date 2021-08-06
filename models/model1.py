import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Build dataset
class DiagnosisDataset(Dataset):
    def __init__(self,data_list, seq_length, pad_value=0 ,drop_len=1):
        self.seq_length = seq_length
        self.pad_value = pad_value
        self.drop_len = drop_len
        
        self.data_list = [x for x in data_list if len(x) > self.drop_len]
        self.input, self.target = self.input_target(self.data_list)
    
    def __getitem__(self,idx):
        inputs = np.array(self.input[idx]).astype(np.int64)
        inputs = padding(inputs,self.seq_length, pad_value=self.pad_value)
        
        targets = np.array(self.target[idx]).astype(np.int64)
        return inputs, targets

    def __len__(self):
        return len(self.target)
    
    def input_target(self,x):
        inputs = []
        targets = []
        for data_ in x:
            len_ = len(data_)
            inputs.append(data_[:(len_-1)])
            targets.append(data_[-1])
        return inputs, targets
    
    
class PatientLSTM(nn.Module):
    def __init__(self,n_feature,n_hidden,n_layer,drop_prob,vocab_size = 2626,padding_val = 2625):
        super().__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.drop_prob = drop_prob
        
        self.embedding = nn.Embedding(vocab_size,self.n_feature,padding_idx=padding_val)
        self.lstm = nn.LSTM(self.n_feature,self.n_hidden,self.n_layer,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(11*self.n_hidden,vocab_size)
        
    def forward(self,x):
        embedded = self.embedding(x)
        lstm_ouput, hidden = self.lstm(embedded)
        out = self.dropout(lstm_ouput)
        out = out.reshape(-1,11*self.n_hidden)
        out = self.fc(out)
        return out, hidden

def padding(x_array, length, pad_value=0):
    """
    x_array: to be padded
    length: max length
    """
    len_ = len(x_array)
    len2pad = length-len_
    assert len2pad >= 0,"padding length should >= the array length"
    
    padded_x = np.pad(x_array,(0,len2pad),mode="constant",constant_values=pad_value)
    return padded_x

    
def train_model(n_feature,n_hidden,n_layer,drop_prob,batch_size,input_size,train,val,num_epoch,pad_value=2625,save_path="save_model/latest_model.pth"): 
    
    trainset = DiagnosisDataset(train,input_size,pad_value=pad_value)
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    valset = DiagnosisDataset(val,input_size,pad_value=pad_value)
    valloader = DataLoader(valset,batch_size=500,shuffle=True)
    #model define
    model = PatientLSTM(n_feature,n_hidden,n_layer,drop_prob)
    model.to(device)
    #define loss
    creteria = nn.CrossEntropyLoss()
    # define optimizer
    optimizer = optim.SGD(model.parameters(),lr=0.02, momentum=0.9)
    print(f"{'epoch':15s}{'train_loss':20s}")
    print("-"*60)
    
    for epoch in range(num_epoch):
    # set model into train  mode
    # h = model.init_hidden(batch_size)
        model.train()
        train_loss = []
        for bt_idx, (inputs,targets) in enumerate(trainloader):
            # set data to device
            inputs, targets = inputs.to(device), targets.to(device)
            # h = tuple([each.data for each in h])
            # make predictions
            output, (_) = model(inputs)
            # compute loss
            tr_loss = creteria(output, targets)
            # set gradients to zero
            model.zero_grad()
            # backpropagate the gradients
            tr_loss.backward()
            train_loss.append(tr_loss.item())
            # upadte the weights
            optimizer.step()
            # scheduler.step(tr_loss.item())

        # set model eval mode
        model.eval()
        test_loss = []
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs,targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                y_pred,_ = model(inputs)
                loss = creteria(y_pred, targets)
                test_loss.append(loss.item())
                # apply softmax to final layer
                y_pred = F.softmax(y_pred, 1).cpu()
                # get max score and index
                score, indx = torch.max(y_pred, 1)
                correct += torch.eq(targets.cpu(),indx).sum()
                total += targets.size()[0]
        print(f"{epoch+1:4d}{np.mean(train_loss):18.4f}")
        if epoch%50==0:
            torch.save(model.state_dict(), f"save_model/model-{epoch}-{np.mean(train_loss):18.4f}.pth")
        f = open("demofile1l.txt", "w")
        f.write(f"{epoch+1:4d}{np.mean(train_loss):18.4f}\n")
        f.close()
    print("Save model..")
    torch.save(model.state_dict(), save_path)
    print("Training finished...")

def load_model(n_feature,n_hidden,n_layer,drop_prob,save_path):
    
#     device = torch.device('cpu')
    model = PatientLSTM(n_feature,n_hidden,n_layer,drop_prob)
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Model Loaded")
    
    return model

def infer(x_test,model):
    
    _x=torch.from_numpy(np.array(x_test)).view(1,-1)
    with torch.no_grad():
        y_hat, _ = model(_x.to(device))
        y_hat = F.softmax(y_hat,1).cpu()
        _, indx = torch.max(y_hat,1)
    return indx.item()