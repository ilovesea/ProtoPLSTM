import csv
import json
import hashlib
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from convlstm_features import convlstm_features
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

class ProtoLayer(nn.Module):
    def __init__(self, n_proto, proto_channels, proto_h, proto_w):
        super(ProtoLayer,self).__init__()
        self.prototypes = nn.Parameter(torch.rand(n_proto, proto_channels, proto_h, proto_w))
        self.n_proto = n_proto
        self.n_channels = proto_channels
        self.hp = proto_h
        self.wp = proto_w

    def forward(self, x):
        ones = torch.ones(self.prototypes.shape).to(x.device)
        x2 = x ** 2
        x2_patch_sum = F.conv2d(x2, ones)
        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim = (1,2,3))
        p2_reshape = p2.view(-1, 1, 1)               
        xp = F.conv2d(x, self.prototypes)
        return F.relu(x2_patch_sum - 2 * xp + p2_reshape)

class AProtoPNet(nn.Module):
    def __init__(self,
                 num_classes=15,
                 num_prototypes=10,
                 prototype_shape=1,
                 use_gcb=True,
                 gcb_dim=324,
                 pool_type="max"):
        super(AProtoPNet, self).__init__()
        self.features = convlstm_features()
        self.use_gcb = use_gcb
        if use_gcb:
            self.gcb = GCB(gcb_dim)
        self.proto_layer = ProtoLayer(
            num_classes * num_prototypes, 128, prototype_shape, prototype_shape)
        self.fc_layer = nn.Linear(num_classes * num_prototypes, num_classes)
        self.pool_type = pool_type
        self.epsilon = 1e-4
        self.proto_info = []

    def pool(self, x):
        if self.pool_type == "max":
            return (-F.max_pool2d(-x, (x.shape[-2],x.shape[-1]))).view(-1,x.shape[1])
        elif self.pool_type == "avg":
            return F.adaptive_avg_pool2d(x, 1).view(-1,x.shape[1])

    def gcb_forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1) # x-> (batch, 128, 324)
        x = self.gcb(x)
        x = x.view(x.size(0), x.size(1), 9, -1) # x-> (batch, 128, 9, 36)
        return x

    def forward(self, x):
        x = self.features(x) # x-> (batch, 128, 9, 36)
        if self.use_gcb:
            x = self.gcb_forward(x)
        x = self.proto_layer(x) # x -> (batch, n_proto, 9 - hp + 1, 11 - hp + 1)
        x = self.pool(x)
        # x -> (batch, n_proto)
        x = self.fc_layer(x) # x -> (batch, 15)
        return x

    def project_prototypes(self, x, y):
        self.proto_info = []
        print("Projecting prototypes...", end="")
        with torch.no_grad():
            x = self.features(x)
            if self.use_gcb:
                x = self.gcb_forward(x)
            proto_out = self.proto_layer(x) # proto_out -> (batch, n_proto, 9 - hp + 1, 11 - hp + 1)
        
            n_samples, n_proto, ho, wo = proto_out.shape 

            for indx_proto in range(n_proto):
                min_dist = np.inf
                if indx_proto == 140:
                    print(f"15")
                elif indx_proto % 10 == 0:
                    print(f"{int(indx_proto/10+1)}..", end="")
                for indx_sample in range(n_samples):
                    for h in range(ho):
                        for w in range(wo):
                            if proto_out[indx_sample,indx_proto, h, w].item() < min_dist:
                                h_min = h
                                w_min = w
                                indx_sample_min = indx_sample
                                min_dist = proto_out[indx_sample, indx_proto, h, w].item()
                hp, wp = self.proto_layer.prototypes.shape[-2], self.proto_layer.prototypes.shape[-1]
                self.proto_info.append((indx_proto, indx_sample_min, y[indx_sample_min].item()))
                self.proto_layer.prototypes.data[indx_proto] = x[indx_sample_min, :, h_min: h_min + hp, w_min: w_min + wp]
 
    def prototype_visualize(self, x, y):
        prototypes = pd.read_csv(Path(__file__).resolve().parent / "saved_models/prototypes.csv", sep=" ")
        with torch.no_grad():
            x_shape = x.shape
            x = self.features(x)
            x = self.gcb_forward(x)
            distances = self.proto_layer(x) # (batch, 150, 9, 36)
            min_distances = (-torch.max_pool2d(-distances, (distances.shape[-2],distances.shape[-1]))).view(-1,distances.shape[1]) # (batch, 150)
            prototype_activations = torch.log10((min_distances + 1) / (min_distances + self.epsilon))
            prototype_activation_patterns = torch.log10((distances + 1) / (distances + self.epsilon))
            upsampled_activation_patterns = []
            for prototype_activation, prototype_activation_pattern, label in zip(prototype_activations, prototype_activation_patterns, y):
                array_act, sorted_indices_act = torch.sort(prototype_activation)
                sorted_indices_act = [i.item() for i in sorted_indices_act if i in prototypes.index[prototypes['label'] == label.item()].tolist()]
                activation_pattern = prototype_activation_pattern[sorted_indices_act].detach().cpu() # most similar prototype
                activation_pattern = activation_pattern.unsqueeze(0)
                upsampled_activation_pattern = F.interpolate(activation_pattern, x_shape[2:]).squeeze(0)
                upsampled_activation_patterns.append((upsampled_activation_pattern, sorted_indices_act, label))

            return upsampled_activation_patterns


def train(model, train_loader: DataLoader, epoch, device, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item() / train_loader.batch_size))

def test(model, test_loader, device, criterion, scheduler):
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
    scheduler.step(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

if __name__ == '__main__':
    config = {
        "dataset": {
            "sample": "3000" # "800"
        },
        "model": {
            "num_prototypes": 10,
            "prototype_shape": 1,
            "use_gcb": True,
            "gcb_dim": 324, # 72
            "pool_type": "max",
        },
        "training": {
            "gpu": "1",
        }
    }
    print(config)
    train_dataset = SiSFallDataset(train=True, expand_dims=True, sample=config["dataset"]["sample"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)
    test_dataset = SiSFallDataset(train=False, expand_dims=True, sample=config["dataset"]["sample"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    device = torch.device(f"cuda:{config['training']['gpu']}")
    
    model = AProtoPNet(num_prototypes=config["model"]["num_prototypes"],
                       prototype_shape=config["model"]["prototype_shape"],
                       use_gcb=config["model"]["use_gcb"],
                       gcb_dim=config["model"]["gcb_dim"],
                       pool_type=config["model"]["pool_type"])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    r = 5
    best_acc = 0
    namehash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    print(f"Model hash: {namehash}")
    save_name = Path(__file__).resolve().parent / f'saved_models/best_model_{namehash}.pt'
    proto_info_name = Path(__file__).resolve().parent / f"saved_models/prototypes_{namehash}.csv"
    for epoch in range(500):
        train(model, train_loader, epoch, device, optimizer, criterion)
        projection = not divmod(epoch, r)[-1] and epoch !=0
        if projection:
            x_all = torch.Tensor(train_dataset[:][0]).to(device)
            y_all = torch.LongTensor(train_dataset[:][1]).to(device)
            model.project_prototypes(x_all, y_all)
        test_acc = test(model, test_loader, device, criterion, scheduler)
        if projection and test_acc > best_acc:
            best_acc = test_acc
            print(f"Current Best Accuracy: {best_acc}")
            torch.save(model.state_dict(), save_name)
            with open(proto_info_name, 'w', newline='') as csvfile:
                protowriter = csv.writer(csvfile, delimiter=' ',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                protowriter.writerow(["prototype", "train_sample", "label"])
                protowriter.writerows(model.proto_info)
            