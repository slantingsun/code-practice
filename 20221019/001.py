import torch
from tqdm import tqdm
import sys


def train_on_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    dataloader = tqdm(dataloader, file=sys.stdout)
    mean_loss = torch.zeros(1).to(device)
    mean_acc = torch.zeros(1).to(device)
    for step, data in enumerate(dataloader):
        optimizer.zero_grad()
        input, label = data
        output = model(input.to(device))
        loss = criterion(output, label.to(device))

        mean_loss = (mean_loss * step + loss.item()) / (step + 1)

        dataloader.desc = f"[Epoch{epoch}] mean loss:{mean_loss.item():.3f}"


        if torch.isfinite(loss):
            print("WARING: no finite loss,end training", loss)
            sys.exit(1)

        loss.backward()
        optimizer.step()
    return mean_loss.item()

def