import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset_sisfall import SiSFallDataset
from protopnet import AProtoPNet
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def test(model, test_loader, device, criterion, scheduler):
    model.eval()
    test_loss = 0
    correct = 0
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            y_pred = torch.cat((y_pred, pred.cpu()), 0)
            y_true = torch.cat((y_true, target.cpu()), 0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    scheduler.step(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset), y_pred, y_true

if __name__ == '__main__':
    test_dataset = SiSFallDataset(train=False, expand_dims=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = AProtoPNet()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load('saved_models/best_model_gcb.pt'))
    test_acc, y_pred, y_true = test(model, test_loader, device, criterion, scheduler)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, range(15), range(15))
    sn.set_theme(context="talk", style="white", palette=None)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu")
    plt.savefig("cm.png")