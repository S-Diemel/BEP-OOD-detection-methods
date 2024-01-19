import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

        
def load_net(name,architecture, path = "checkpoint/"):
    checkpoint = torch.load(path+name,map_location='cpu')
    architecture.load_state_dict(checkpoint['net'])
    architecture.eval()
    print(name+' ACC:\t',checkpoint['acc'])
    return architecture

#get the embeddings for all data points
def gather_embeddings(net, d, loader: torch.utils.data.DataLoader, device, storage_device):
    num_samples = len(loader.dataset)
    output = torch.empty((num_samples, d), dtype=torch.double, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    confs = []
    with torch.no_grad():
        start = 0

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            out = net.embed(data)


            end = start + len(data)
            output[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end
            conf = net.conf(data)
            for probs in conf:
                confs.append(probs)
    return output, labels, confs

def gather_embeddings_last(net, d, loader: torch.utils.data.DataLoader, device, storage_device):
    num_samples = len(loader.dataset)
    output = torch.empty((num_samples, d), dtype=torch.double, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    confs = []
    outs = []
    with torch.no_grad():
        start = 0

        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device), label.to(device)

            out = net(data)


            end = start + len(data)
            output[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end
            conf = net.conf(data)
            for probs in conf:
                probs=probs.to(torch.float64)
                confs.append(probs)
            for ut in out:
                ut=ut.to(torch.float64)
                outs.append(ut)
        print(len(outs))
        outs = torch.stack(outs)
        confs = torch.stack(confs)
        for i in range(len(loader.dataset)):
            if outs[i][0]!=output[i][0]:
                print(i, 'error')

    return outs, labels, confs

def gather_embeddings_odin(net, d, loader: torch.utils.data.DataLoader, device, storage_device, epsilon, temper, criterion):
    num_samples = len(loader.dataset)
    output = torch.empty((num_samples, d), dtype=torch.double, device=storage_device)
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)
    confs = []

    start = 0

    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        inputs = Variable(data.cuda(), requires_grad=True)
        outputs = net.forward(inputs)
        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        all_labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        outputs = outputs / temper
        loss = criterion(outputs, all_labels)
        loss.backward()
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data, -epsilon, gradient)
        with torch.no_grad():
            out = net.embed(tempInputs).to('cpu')


            end = start + len(data)
            output[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end
            conf = net.conf(data).to('cpu')
            for probs in conf:
                confs.append(probs)
    return output, labels, confs