import torch
from tqdm import tqdm
import loss
import lpips_loss
from net import KenYaNet
import torch.optim as optim
from dataset import LOLPairImagesDataset
from torch.utils.data import DataLoader

net = KenYaNet()
net = net.cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

lol_dataset = LOLPairImagesDataset(lol_path='E:\LOL-v2\Real_captured\Train')
lol_dataloader = DataLoader(lol_dataset, batch_size=8, shuffle=True, num_workers=1)

my_loss = loss.MyLoss()

epoch = 0

best_loss = 1e10


def train():
    net.train()
    loss_print = 0
    tqdm_bar = tqdm(lol_dataloader)
    for iteration, batch in enumerate(tqdm_bar, 1):
        original_img, normal_img = batch[0], batch[1]
        original_img = original_img.cuda()
        normal_img = normal_img.cuda()
        enhance_img = net(original_img)
        delta = normal_img - original_img
        _loss = torch.nn.functional.mse_loss(delta, enhance_img)
        optimizer.zero_grad()
        _loss.backward()
        optimizer.step()
        loss_print = loss_print + _loss.item()
        if iteration % 10 == 0:
            tqdm_bar.set_description("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                                                                                                           iteration,
                                                                                                           len(
                                                                                                               lol_dataloader),
                                                                                                           loss_print,
                                                                                                           optimizer.param_groups[
                                                                                                               0][
                                                                                                               'lr']))
            loss_print = 0
    return loss_print


if __name__ == '__main__':
    for i in range(50):
        train_loss = train()
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(net.state_dict(), 'checkpoint.pth')
            print('model saved.')
        epoch += 1
