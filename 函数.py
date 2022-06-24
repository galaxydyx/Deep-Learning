import torch
import torchvision.transforms as transforms
import tqdm
import sys
import random


def min_max_scaler(dat):
    mi = torch.min(dat)
    ma = torch.max(dat)
    return (dat - mi)/(ma-mi)


class FlameSet(torch.utils.data.Dataset):
    def __init__(self, need_data_len, data, bite, label_data):
        super(FlameSet, self).__init__()
        # random.seed(0)
        # rand = random.sample(range(data_len), need_data_len)
        self.files = []
        # z-score规范化
        a_ave = 0
        a_temp = 0
        r, g, b = data[0].size()
        for k in range(need_data_len):
            img_file = data[k]
            label_file = label_data[k]#%10000[0].reshape(g, b)#torch.tensor().type(torch.FloatTensor)
            a_ave += torch.mean(img_file)
            for i in range(bite):
                label_file[(label_file > i/bite) & (label_file <= (i+1)/bite)] = i
            label_file = label_file.type(torch.LongTensor)
            self.files.append({
                "img": img_file,
                "label": label_file})
        a_ave = a_ave / need_data_len
        for k in range(need_data_len):
            a_temp += torch.sum((data[k] - a_ave) ** 2) / (r * g * b)
        a_temp = (a_temp / need_data_len) ** 0.5
        self.transforms_a = transforms.Normalize((a_ave,), (a_temp,))
        self.transforms_b = transforms.Normalize((.5,), (.5,))

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = datafiles["img"]
        label = datafiles["label"]
        img = self.transforms_a(image)
        img = min_max_scaler(img)
        img = self.transforms_b(img)
        # label = self.transforms_b(label)
        return img, label

    def __len__(self):
        return len(self.files)


def collate_fn(batch):
    img, lab = tuple(zip(*batch))
    img = torch.stack(img, dim=0)
    lab = torch.stack(lab, dim=0)
    return img, lab


def train_one_epoch(model, optimizer, loss_fn, data_loader, device, epoch):
    model.train()
    loss_fn = loss_fn
    batch_loss = []
    mean_loss = torch.zeros(1).to(device)
#     if dist.get_rank()==0:
    data_loader = tqdm.tqdm(data_loader)
    for i, data_i in enumerate(data_loader):
        inputs, labels = data_i
        outputs = model(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))###
        optimizer.zero_grad()
        loss.backward()
        batch_loss.append(loss.detach())
#         loss = reduce_value(loss,average=True)#多gpu使用
        mean_loss = (mean_loss*i+loss.detach())/(i+1)
#         if dist.get_rank()==0:
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
        if not torch.isfinite(loss):
            print("nan-finite loss, ending training,loss is {}".format(loss))
            sys.exit(1)
        optimizer.step()
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return mean_loss, batch_loss


@torch.no_grad()
def test(model, loss_fn, data_loader, device):
    model.eval()
    output = []
    batch_loss = []
    mean_loss = torch.zeros(1).to(device)
    loss_fn = loss_fn
    #     if dist.get_rank()==0:
    data_loader = tqdm.tqdm(data_loader)
    for i, data in enumerate(data_loader):
        images, labels = data
        outputs = model(images.to(device))
        output.append(outputs)
        loss_test = loss_fn(outputs, labels.to(device))
        mean_loss = (mean_loss * i + loss_test) / (i + 1)
        data_loader.desc = "mean loss {}".format(round(mean_loss.item(), 3))
        batch_loss.append(loss_test)
    if device != "cpu":
        torch.cuda.synchronize(device)
    return mean_loss, batch_loss, output



