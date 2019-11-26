import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=False, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=False, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)

model = torch.load('victim.pt')
gaussian = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5]))

#_______________________________NOGRAD NOGRAD NOGRAD_____________________________________________############


print("Without gradient queries")

attacker = nn.Sequential(nn.Linear(784,64),nn.ReLU(),nn.Linear(64, 10),nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.SGD(attacker.parameters(), lr=0.001, momentum=0.9)

count = 0
for images, labels in valloader:
	#images = images.view(images.shape[0], -1)
	images = gaussian.sample((1,784))
	images = images.view(images.shape[0], -1)
	optimizer.zero_grad()
	output = attacker(images)
	pr = list(torch.exp(model(images)).detach().numpy()[0])
	lab = torch.Tensor([(pr.index(max(pr)))]).to(torch.long)
	# print(labels)
	# print(lab)
	# print(pr.index(max(pr)))
	# break
	loss = criterion(output, lab)
	loss.backward()
	optimizer.step()

	#running_loss += loss.item()
	if count == 999 or count == 1999 or count == 4999 or count == 9999:
		print("Epoch {} - Training loss: {}".format(count, loss.item()))
		correct = 0
		total = 0
		
		for images1,labels1 in valloader:
			for i in range(len(labels1)):
				img = images1[i].view(1, 784)
				with torch.no_grad():
					logps = attacker(img)

			ps = torch.exp(logps)
			probab = list(ps.numpy()[0])
			pred_label = probab.index(max(probab))
			true_label = labels1.numpy()[i]
			if(true_label == pred_label):
				correct += 1
			total += 1
		print("Accuracy =", (correct/total))
	count += 1








#______________GRAD GRAD GRAD GRAD ________________#########################################################



print("\nWith gradient queries")


attacker = nn.Sequential(nn.Linear(784,64),nn.ReLU(),nn.Linear(64, 10),nn.LogSoftmax(dim=1))
criterion1 = nn.NLLLoss()
optimizer = optim.SGD(attacker.parameters(), lr=0.001, momentum=0.9)

count = 0
for images, labels in valloader:

	#images = images.view(images.shape[0], -1)
	images = gaussian.sample((1,784))
	images = images.view(images.shape[0], -1)
	imagesV = images.clone().detach()
	imagesA = images.clone().detach()
	imagesC = images.clone().detach()

	images.requires_grad = True
	imagesA.requires_grad = True
	imagesV.requires_grad = True

	output = attacker(images)
	outputA = attacker(imagesA)
	outputV = model(imagesV)
	outputC = model(imagesC)

	mseLoss = torch.zeros(1,requires_grad=True)

	pr = list(torch.exp(outputC).detach().numpy()[0])
	lab = torch.Tensor([(pr.index(max(pr)))]).to(torch.long)

	for i in range(10):
		bol = True
		if i == 9:
			bol = False
		outputA[0,i].backward(retain_graph = bol)
		gradAttackerT = imagesA.grad
		outputV[0,i].backward(retain_graph = bol)
		gradVictimT = imagesV.grad
		mseLoss = mseLoss + torch.matmul(gradVictimT - gradAttackerT,torch.t(gradVictimT - gradAttackerT))
	mseLoss = mseLoss / 7840

	optimizer.zero_grad()

	# print(labels)
	# print(lab)
	# print(pr.index(max(pr)))
	# print(mseLoss)
	# break
	#print(mseLoss)

	loss = criterion1(output, lab) + mseLoss 
	loss.backward()
	optimizer.step()
	#running_loss += loss.item()
	if count == 999 or count == 1999 or count == 4999 or count == 9999:
		print("Epoch {} - Training loss: {} , {}".format(count, loss.item(), mseLoss))
		correct = 0
		total = 0
		for images1,labels1 in valloader:
			for i in range(len(labels1)):
				img = images1[i].view(1, 784)
				with torch.no_grad():
					logps = attacker(img)

			ps = torch.exp(logps)
			probab = list(ps.numpy()[0])
			pred_label = probab.index(max(probab))
			true_label = labels1.numpy()[i]
			if(true_label == pred_label):
				correct += 1
			total += 1
		print("Accuracy =", (correct/total))
	count += 1