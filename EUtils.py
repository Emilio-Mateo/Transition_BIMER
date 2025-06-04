import torch
import numpy as np

def validation(model,data,device):
        val = model.eval()
        total = 0
        correct = 0
        for i, (img,label) in enumerate(data):
            with torch.no_grad():
                img = img.to(device)
                x = val(img)
                _, pred = torch.max(x,1)
                pred = pred.to('cpu')
                total += img.size(0)
                correct += torch.sum(pred == label)
                acc = float(correct * 100 /total)
        return acc

def norm_array(x):
        x = x.astype(np.float64)
        l, _ = x.shape
        min = np.min(x, axis=0)
        x -= np.array([min,]*l)
        max = np.max(x, axis=0)
        x /= np.array([max,]*l)
        return x

def norm_ima(x, dtp = None):
    p,q = x.shape
    x -= np.array([np.min(x, axis=0)]*p)
    x /= np.array([np.max(x, axis=0)]*p)
    x = x * 256
    if dtp:
        return x.astype(np.uint8)
    else:
        return x / 256
