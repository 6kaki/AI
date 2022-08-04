import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets


class VGG(nn.Module):
    def __init__(self,vgg_layers,output_dim):
        super().__init__()
        self.vgg_layers = vgg_layers
        self.avg_pool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,output_dim)
        )
    def forward(self,x):
        x = self.vgg_layers(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x


vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
         'M', 512, 512, 512, 'M']
def get_vgg_layer(vgg,batch_norm):
    start_channel = 3
    layers = []

    for c in vgg:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=(2,2),stride=2)]

        else:
            conv2d = nn.Conv2d(start_channel,c,kernel_size=(3,3),padding=1,stride=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(c),nn.ReLU()]
            else:
                layers += [conv2d,nn.ReLU()]
            start_channel = c
    return nn.Sequential(*layers)


if __name__ == '__main__':

    from data import CustomDataset
    from torch.utils.data import random_split,DataLoader
    import data

    vgg16_layer = get_vgg_layer(vgg16,batch_norm=True)
    print(vgg16_layer)
    output_dim = 2
    model = VGG(vgg16_layer,output_dim)
    print(model)

    transforms = transforms.Compose([
        transforms.Resize((256,256)),
        data.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    # test_transforms = transforms.Compose([
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    # ])

    datasets = CustomDataset(size=256,transform=transforms)
    dataset_size = len(datasets)
    train_size = int(dataset_size * 0.8)
    val_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        datasets,
        [train_size,val_size,test_size]
    )

    train_dataloader = DataLoader(train_dataset,
                               shuffle=True,
                               batch_size=4,
                                  collate_fn=lambda x:x)
    val_dataloader = DataLoader(val_dataset,
                               shuffle=True,
                               batch_size=4,
                                  collate_fn=lambda x:x)
    test_dataloader = DataLoader(test_dataset,
                               shuffle=True,
                               batch_size=4,
                                  collate_fn=lambda x:x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    optimizer = optim.Adam(model.parameters(),lr=1e-7)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def calcuate_accuracy(y_hat,y):
        y_hat = y_hat.argmax(1,keepdim=True)
        correct = y_hat.eq(y.view_as(y_hat)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train(model,iter,optimizer,criterion,device):
        epoch_loss=0
        epoch_acc = 0

        model.train()

        for batch in iter:
            for i in batch:
                X = i['image']
                X = torch.FloatTensor(X).to(device)
                y = i['target'][0]['labels']
                y = torch.Tensor(y).to(device)

                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat,y)
                acc = calcuate_accuracy(y_hat,y)
                loss.backward()
                optimizer.step()

            epoch_acc += acc.item()
            epoch_loss += loss.item()
        return epoch_loss / len(iter), epoch_acc / len(iter)

    def eval(model,iter,criterion,device):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()
        with torch.no_gard():
            for batch in iter:
                for i in batch:
                    x = i['image']
                    x = torch.FloatTensor(x).to(device)
                    y = i['target'][0]['labels']
                    y = torch.Tensor(y).to(device)
                    y_hat = model(x)
                    loss = criterion(y_hat,y)
                    acc = calcuate_accuracy(y_hat,y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            return epoch_loss / len(iter), epoch_acc / len(iter)

    def epoch_time(start,end):
        elapsed_time = end - start
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins,elapsed_secs

    epochs = 5
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        start = time.monotonic_ns()
        train_loss,train_acc = train(model,train_dataloader,optimizer,criterion,device)
        valid_loss,valid_acc = eval(model,val_dataloader,criterion,device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),'../VGG-model.pt')

        end = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start,end)

        print(f'Epoch : {epoch + 1:02} | Epoch Time : {epoch_mins}m {epoch_secs}s')
        print(f'\t Train Loss : {train_loss:.3f} | Train Acc : {train_acc * 100:.2f}%')
        print(f'\t Valid Loss : {valid_loss:.3f} | Train Acc : {valid_acc * 100:.2f}%')