import random
import torch
import numpy as np
import scipy.stats as st
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time
import argparse
import sys

device = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500, help="total number of epochs")
parser.add_argument("--reconstruction-param", type=float, default=0.1, help="reconstruction parameter")
parser.add_argument("--entropy-param", type=float, default=1., help="entropy_param")
parser.add_argument("--embedding", type=int, default=100, help="embedding")
parser.add_argument("--num-classes", type=int, default=2, help="number of classes")
args = parser.parse_args()

num_classes = args.num_classes
print(args)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, num_classes=num_classes, embedding = 100):
        super(ResNet, self).__init__()
        self.in_planes = feature_maps

        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(3, feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb, stride = 1 if i == 0 else 2))            
            self.layers = nn.Sequential(*layers)
#        self.embedding = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps * block.expansion, 1) 
        self.linear = nn.Linear(1, num_classes)
        self.depth = len(num_blocks)
        self.bounds = nn.Parameter(torch.Tensor([0.,1.]))

        ### Quantifier dictionary
        self.embd = nn.Embedding(embedding,1)
        self.embd.weight.data.uniform_(-1/embedding, 1/embedding)
        self.MSE_Loss = torch.nn.MSELoss()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, train = False):

        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
        features = features.mean(dim = 1, keepdim = True)
        out = self.linear(features)
        if train: ### We update the dictionary 
            embedding_size = self.embd.weight.size(1)
            codebook_sqr = torch.sum(self.embd.weight ** 2, dim=1)
            inputs_flatten  = features.view(-1, embedding_size)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            
            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr.detach() + inputs_sqr,
            inputs_flatten, self.embd.weight.t().detach(), alpha=-2.0, beta=1.0)
            
            ###Soft one hot sampling
            soft_one_hot = F.gumbel_softmax(-distances) 


            features_quant = torch.einsum('b n, n d -> b d', soft_one_hot, self.embd.weight)

            ###Reconstruction loss 
            loss_quant = self.MSE_Loss(features_quant.detach(),features) * args.reconstruction_param + self.MSE_Loss(features_quant,features.detach()) 

        #logits += torch.rand(logits.size())
            logits = soft_one_hot.sum(0)
            logits =  F.softmax(logits,dim=-1)
        
        #uniform_sample = torch.ones(10)*1/10#self.uniform.rsample(sample_shape = [ 10] )
        #entropy = self.KL_Loss(logits,uniform_sample)
            entropy = - (logits*logits.log()).sum()
            #features = features + (features_quant-features).detach()
            return features, loss_quant, entropy
        else:
            return features

last_update = 0

def train(model, train_loader, optimizer, thresholds, epoch, mixup = "None"):
    model.train()
    global last_update
    accuracy, total_loss, total_loss2, total_elts = 0., 0., 0., 0
    for batch_idx, (images, target, gender) in enumerate(train_loader):
        data, ages = images.to(device), target.to(device)
        target = torch.zeros_like(ages)
        for i in range(len(thresholds)):
            target[torch.where(ages > thresholds[i])[0]] = i + 1

        optimizer.zero_grad()

        features,loss_quant, entropy = model(data,train= True ) # /!\ changed here 
        output = model.linear(features)

        loss = criterion(output, target.long()).mean()
        total_loss += loss.item() * data.shape[0]
        loss += loss_quant -entropy *args.entropy_param

        loss.backward()
        
        total_loss2 += entropy *args.entropy_param * data.shape[0]
        total_elts += data.shape[0]
        optimizer.step()
        if time.time() - last_update > 0.1:
            print("\r{:5d}/{:5d} loss CE: {:.5f}, loss E: {:.5f}".format(batch_idx + 1, len(train_loader), total_loss / total_elts, total_loss2 / total_elts), end = "")
            last_update = time.time()

    return { "train_losses" : [total_loss / total_elts, total_loss2 / total_elts] }


full_features = torch.zeros(0, 10000, 3) # dims are epoch / samples / age + feature + predicted class

def test(model, test_loader, thresholds, epoch):
    global full_features
    model.eval()
    test_loss, accuracy, total_elts = 0, 0, 0
    all_ages = []
    all_features = []
    all_preds = []
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, age = data.to(device), target.to(device)
            target = torch.zeros_like(age)
            for i in range(len(thresholds)):
                target[torch.where(age > thresholds[i])[0]] = i + 1

            features = model(data)
            output = model.linear(features)

            all_ages.append(age)
            all_features.append(features.reshape(features.shape[0]))
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            all_preds.append(pred.reshape(pred.shape[0]))
            accuracy += pred.eq(target.view_as(pred)).sum().item()
            total_elts += target.shape[0]
            if total_elts >= 10000:
                break
            
    ages = torch.cat(all_ages).to("cpu") # all the ages associated with the samples
    features = torch.cat(all_features).to("cpu") # all the features
    preds = torch.cat(all_preds).to("cpu") # all the predictions

    ### just for saving...
    cat_elements = torch.stack([ages, features, preds], dim = 1).unsqueeze(0)
    full_features = torch.cat([full_features, cat_elements], dim = 0)
    torch.save(full_features, "features.pt")

    ### first we want to determine if the features are sorting in the same order as the ages, or the reverse order...
    ### to do that, we order the samples in ascending ages, and look at the average of youngs and olds
    ordered_ages, index_ages = torch.sort(ages)

    first_features = features[index_ages][:features.shape[0] // 2]
    last_features = features[index_ages][features.shape[0] // 2 :]
    if first_features.mean(dim = 0) > last_features.mean(dim = 0):
        features = torch.max(features) - features
    ### now the features are in the same order as the ages

    ### we can try to see if samples are ordered the same when looking at features
    ordered_features, index_features = torch.sort(features)
    MSE = (ages[index_features] - ordered_ages).pow(2).mean().item()

    ### a natural comparison point would be a random permutation
    randmse = (ages[torch.randperm(ages.shape[0])] - ordered_ages).pow(2).mean().item()

    ### next we want to look at the quality of predicting the ages based on the prediction
    ### we associate each class with the corresponding average on the dataset
    nt = [-1] + thresholds + [1000]
    pred_ages = torch.zeros_like(ages)
    hard_ages = torch.zeros_like(ages)
    for i in range(len(nt) - 1):
        indices = torch.where((ordered_ages > nt[i]) * (ordered_ages <= nt[i+1]))[0]
        avg = ordered_ages[indices].mean()
        pred_ages[torch.where(preds[index_ages] == i)[0]] = avg
        hard_ages[indices] = avg
    classif_mse = (pred_ages - ordered_ages).pow(2).mean().item()
    hard_mse = (hard_ages - ordered_ages).pow(2).mean().item()

    ### finally, we look at an interpolation between both
    inter_mse, inter_hard_mse = 1000000., 1000000.
    best_coeff, best_hard_coeff = -1, -1
    for beta in np.linspace(0, 1, 101):
        new_mse = ((beta * pred_ages + (1 - beta) * ages[index_features]) - ordered_ages).pow(2).mean().item()
        new_hard_mse = ((beta * hard_ages + (1 - beta) * ages[index_features]) - ordered_ages).pow(2).mean().item()
        if new_mse < inter_mse:
            best_coeff = beta
            inter_mse = new_mse
        if new_hard_mse < inter_hard_mse:
            best_hard_coeff = beta
            inter_hard_mse = new_hard_mse

    return { "test_loss" : test_loss / (batch_idx + 1), "test_acc" : accuracy / total_elts , "mse" : MSE, "randmse": randmse, "classifmse": classif_mse, "intermse": inter_mse, "inter_coeff": best_coeff, "hardmse": hard_mse, "interhardmse": inter_hard_mse, "inter_hard_coeff": best_hard_coeff }

def train_era(model, offset_epochs, epochs, lr, loaders, thresholds, mixup = False, verbose = True):
    if lr < 0:
        optimizer = torch.optim.Adam(model.parameters())
    else:
        all_params = set(model.parameters())
        wd_params = set()
        for m in model.modules():
            try:
                _ = m.weight.shape
                wd_params.add(m.weight)
            except:
                pass
        no_wd = all_params - wd_params
        optimizer = torch.optim.SGD([{'params':list(wd_params)}, {'params':list(no_wd), 'weight_decay':0}], lr = lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
    train_loader, test_loader = loaders
    for epoch in range(offset_epochs, offset_epochs + epochs):
        train_stats = train(model, train_loader, optimizer, thresholds, epoch, mixup = mixup)
        test_stats = test(model, test_loader, thresholds, epoch)
        print("\rEpoch: {:3d}, test_acc: {:.4f}%, train_losses: {:.5f}, {:.5f}, MSE: {:.2f}, rand: {:.2f}, classif: {:.2f}, inter: {:.2f}, hard: {:.2f}, interhard: {:.2f}".format(epoch, 100*test_stats["test_acc"], train_stats["train_losses"][0], train_stats["train_losses"][1], test_stats["mse"], test_stats["randmse"], test_stats["classifmse"], test_stats["intermse"], test_stats["hardmse"], test_stats["interhardmse"]))
        f = open("results_" +str(args.entropy_param)+"_"+str(args.entropy_param)+"_"+str(args.embedding)+ ".csv", "a")
        f.write("{:4d}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(epoch, test_stats["test_acc"], train_stats["train_losses"][0], train_stats["train_losses"][1], test_stats["mse"], test_stats["randmse"], test_stats["classifmse"], test_stats["intermse"], test_stats["inter_coeff"], test_stats["hardmse"], test_stats["interhardmse"], test_stats["inter_hard_coeff"]))
        f.close()

    print()
    return train_stats, test_stats

def train_complete(model, training, loaders, thresholds, mixup = False):
    global start_time
    f = open("results_"+str(args.entropy_param)+"_"+str(args.entropy_param)+"_"+str(args.embedding)+".csv","w")
    f.write("epoch, test_acc, ce_loss, entropy_loss, MSE_features, MSE_rand, MSE_classif, MSE_inter, inter_coeff, MSE_hard, MSE_inter_hard, inter_hard_coeff\n")
    f.close()
    start_time = time.time()
    offset_epochs = 0
    for (epochs, lr) in training:
        train_stats, test_stats = train_era(model, offset_epochs, epochs, lr, loaders, thresholds, mixup = mixup)
        f = open("results_" +str(args.entropy_param)+"_"+str(args.entropy_param)+"_"+str(args.embedding)+ ".csv", "a")
        print(test_stats["test_acc"])
        offset_epochs += epochs
    return test_acc

from datasets import MyDatasets
from torchvision import transforms
transformations = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = MyDatasets.IMDBWIKI('../data/imdb_crop', '../data/imdb_crop/imdbfilelist.txt', transformations, db='imdb')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 100, shuffle = False, num_workers = 8)

print("computing thresholds ", end='')
meta_data = open("../data/imdb_crop/imdbfilelist.txt")
all_vals = []
try:
    while True:
        line = meta_data.readline()
        vals = line.split("\t")
        try:
            all_vals.append(int(vals[1]))
        except:
            exit()
except:
    meta_data.close()
print("using " + str(len(all_vals)) + " data elements ", end='')
all_vals = torch.Tensor(all_vals)
sorted_values = torch.sort(all_vals)[0]
thresholds = [sorted_values[(i+1) * sorted_values.shape[0] // num_classes].item() for i in range(num_classes - 1)]
print("done")
print(thresholds)

nt = [-1] + thresholds + [1000]
ref_mse = 0
for i in range(len(nt) - 1):
    vals = all_vals[torch.where((all_vals > nt[i]) * (all_vals <= nt[i+1]))[0]]
    ref_mse += torch.var(vals).item()
print("ref MSE: " + str(ref_mse / num_classes))

#SGD
training = [(30, 0.1),(30,0.01),(30,0.001)]
#Adam
training = [(500, -0.001)]

scores = []

criterion = nn.CrossEntropyLoss()
#def criterion(output, target):
#    target_one_hot = torch.nn.functional.one_hot(target, num_classes = num_classes).float()
#    return nn.MSELoss()(output, target_one_hot)


model = ResNet(BasicBlock, [2, 2, 2, 2], 16).to(device)
train_complete(model, training, (train_loader, test_loader), thresholds)

