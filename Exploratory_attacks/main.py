import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=True, transform=transform) # MAKE DOWNLOAD = TRUE FIRST TIME
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=False, train=False, transform=transform)	# MAKE DOWNLOAD = TRUE FIRST TIME
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


model = nn.Sequential(nn.Linear(784,64),nn.ReLU(),nn.Linear(64, 10),nn.LogSoftmax(dim=1))

print(model)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 15

for e in range(epochs):
	running_loss = 0
	for images, labels in trainloader:
		images = images.view(images.shape[0], -1)
		optimizer.zero_grad()
		output = model(images)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
	else:
		print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))


correct = 0
total = 0
for images,labels in valloader:
	for i in range(len(labels)):
		img = images[i].view(1, 784)
		with torch.no_grad():
			logps = model(img)

	ps = torch.exp(logps)
	probab = list(ps.numpy()[0])
	pred_label = probab.index(max(probab))
	true_label = labels.numpy()[i]
	if(true_label == pred_label):
		correct += 1
	total += 1

print("\nAccuracy =", (correct/total))

torch.save(model, './victim.pt') 
