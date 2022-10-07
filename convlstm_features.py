from torch import nn
import torch
from torch.optim import Adam
from dataset_sisfall import SiSFallDataset

class GCB(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, dim, patch_dim=49):
        super(GCB, self).__init__()

        exp_dim = int(dim * 1.)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, patches):
        h = patches
        x = self.cm(patches)
        x = torch.bmm(h.permute(0, 2, 1), F.softmax(x, 1)).squeeze(-1)
        x = self.wv1(x)
        x = self.gelu(self.norm(x))
        x = self.wv2(x)
        x = h + x.unsqueeze(1)
        x = self.ffn_norm(x)
        x = torch.sigmoid(x)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, filters, filter_size, num_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, filters[0], (1, filter_size), stride=(1, 3))
        self.conv2 = nn.Conv2d(filters[0], filters[1], (1, filter_size), stride=(1, 3))
        self.conv3 = nn.Conv2d(filters[1], filters[2], (1, filter_size), stride=(1, 3))
        self.conv4 = nn.Conv2d(filters[2], filters[3], (1, filter_size), stride=(1, 3))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.rnn = nn.GRU(
            filters[3] * 9,
            int(filters[3] / 4 * 9),
            2,
            bidirectional=True,
            dropout=0.4,
        )
        
        self.num_classes = num_classes
        if num_classes > 0:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(int(filters[3] / 2), num_classes)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        x = x.view(x.size(0), -1, x.size(-1))
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)

        x = x.permute(1, 2, 0)
        x = x.view(x.size(0), -1, 9, x.size(-1))
        

        if self.num_classes:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def convlstm_features(**kwargs):
    return ConvLSTM([32, 64, 128, 256], 5, **kwargs)

def train(model, train_loader, epoch, device, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    train_dataset = SiSFallDataset(train=True, expand_dims=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)
    test_dataset = SiSFallDataset(train=False, expand_dims=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = convlstm_features(num_classes=15)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(100):
        train(model, train_loader, epoch, device, optimizer, criterion)
        test(model, test_loader, device, criterion)