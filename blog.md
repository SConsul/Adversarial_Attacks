

# Adversarial Attacks 


Deep learning has progressed exponentially and has allowed machines to perform tasks that far exceed human performance. However there are vulnerabilities in these models that can be exploited, rendering them extremely brittle. A whole new subfield in Deep learning has opened up which works towards finding possible attacks and developing full-proof defenses against the adversaries. This field is formally called "Adversarial Machine Learning" and has made important contributions to machine learning security. The aim of our blog is to demonstrate 3 contrastingly different adversarial attacks on neural networks, develop an intuition of the how they work and do an analysis of their severity under different settings.
Adversarial attacks can broadly be divided into:

- Poisoning attacks
- Evasion attacks
- Exploratory attacks

We describe them one by one below:



## Poisoning Attacks:

Adversarial poisoning attacks involves the injection of "poisoned" examples in the training set which manipulates the test time performance of a model. These attacks are usually *targeted* i.e. they are done to manipulate the performance of the model on specific test examples. Poisoning attacks only  require an access to the model parameters without the need of other training data. Clean label attacks are a specific type of poisoning attacks that don't require the control of the labelling process while building the train dataset. These attacks look seemingly innocuous and can go unnoticed even by an expert labelling entity. Hence this opens the door for attackers to succeed without any inside access to the data collection/labeling process. An adversary could craft poison examples and leave them online to be scrapped by a data collecting bot. For example, a neural network in a self driving car could be trained with poisoned instances to mis-classify a "STOP sign" to another instuction. 

To demonstrate poisoning attacks we have implemented the paper "Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks"  by Shafahi et. al [1]. This paper gives a powerful technique to execute Clean-label techniques that don't compromise on the test-time performance and miss-classify only a specific target instance for which the poison examples are crafted. Next we described the approach used in the paper to generate the poisoned data. 

### Executing  Simple Clean Label attacks:
The paper uses a simple approach to craft posion instances. The adversary here has access to the model parameters.  The adversary first selects a base class which is used for generating the malicious examples.  The adversary then selects a target instance which it wants to miss-classify. A base instance is taken from the base class and imperceptible changes are made to this using an optimization based procedure to give rise to the poison instances. The poison instances are assigned the labels of the base class. The model is retrained upon the original training data+ poisoned instances. If the  target is successfully miss-classified to the base class then the poisoning attack is said to be successful. 

#### Crafting Poison instances through feature collisions:

The procedure described above effectively boils down to creating a poison instance such that it perceptually appears to be from the base class while it collides with the target instance in the feature space. By feature space we mean the highest level of features extracted by the classifier just before applying a softmax layer. Let $f$ be the feature space of the model. Hence we want a poison instance **p** from the base class **b** for a target **t** such that:
$$ \textbf{p} = \argmin_{x} || f(x) -f(t) ||_{2}^{2} + \beta || x-b||_{2}^{2} \ $$
The first term on the RHS forces the features of the poisoning instance to collide with the features of the target and the second term forces the poisoning instance to be closer to the base class perceptually. $\beta$ decides which term dominates in the objective. Now the poison instance labelled as the base class is located extremely close to the target instance in the feature space.  Hence training over the poisoning instances will inadvertently cause the linear decision boundary to cross over the target instance and indvertently classify it as the base class. Hence this gives an adversary a *backdoor* into the model. 

#### Algorithm for crafting poison instances

Following is a iterative procedure to optimize the objective given previously:

---
**Input**$\leftarrow$target instance **t**, base class **b**,  learning rate $\lambda$
**Initialize** x:  $x_{0}\leftarrow b$ 
Define $L_{p}(x) = || f(x)- f(t) ||^{2}$
for i in 1 to total_iterations:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Forward Step: $\hat{x_{i}} = x_{i-1} - \lambda \nabla L_{p}(x_{i-1})$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Backward Step: $x_{i} = \frac{\hat{x_{i}} + \lambda \beta b}{1+\lambda \beta}$

end for

Watermark x with t

---

The fowards step essentially causes a feature collision between the target and the poison instance. The forward step reduces the Frobenius distance between the poison instance and base class. Tuning $\beta$ influences how close the posioned instance looks to the base instance. Watermarking is used to strengthen the attacks. 


![Decision Boundary](images/decision_b.png)
<p align="center"><em>Decision boundary rotation in a clean label attack</em></p>

### One Shot Kill attacks on Transfer Learning tasks

Transfer learning tasks are extremely vulnerable to clean label attacks. Infact even a single poisoned instance is capable of breaking the neural network during test time. It is common to use pre-trained networks as feature extractors with additonal layers added for a task. Popular architectures are used as pretrained networks and an adversary may have easy access to the model parameters. Here we use the CNN of  **Alex-Net** as a feature extractor and it is frozen during training. The dense network of Alex-Net is discarded.  

![Alex Net](images/alex_net.png)
<p align="center"><em>Alex-Net Architecture, We retain only the CNN portion</em></p>

A final layer is added and is the only trainable part of the model. The model is trained to classify between instances of airplanes and automobiles. We use 800 training images (having equal number of airplanes and automobiles) and 100 test images. Automobiles are chosen as the base class and a particular instance is used as the base instance. We run the iterative optimization algorithm twice for every test instance, with and without the watermarking step. Our hyperparameters are set as follows:

$\beta=16, \lambda = 10^{-4}$, Opacity for Watermarking =0.15, Total iterations = 1000
 
 We obtain the following results:

 Success rate without watermarking = 61%

 Success rate with watermarking = 96%

The performance of the classifier is not compromised while injecting poison instances. The classification accuracy difference before and after poisoning is negligible and hence these attacks can go easily unnoticed. 
Test accuracy of the classifier before poisoning = 97.4%
Average test accuracy of the classifier after poisoning = 97.1%

Hence watermarking increases the intensity of the attacks. Moreover the task is done only over a single poisoned instance and multiple instances can further exacerbate the severity of the attacks. We show diferent examples of the base, target and poisoned instances below. Note how seemingly innocuous the poison instances look and can never be suspected upon by an expert labeller. 

![Clean Label attacks for different target instances](images/Poison.png)
<p align="center"><em>Base, Target and Poisoned Instances in a one shot kill attack</em></p>


### Poisoning attacks on end-to-end Training 
We investigate the effectiveness of the attacks in an end to end training scenario. For this we use a custom CNN architecture. The architecture has 2 conv layers and 2 fully connected layers and is shown below:

![e2e arch](images/e2e.png)
<p align="center"><em>Architecture used for end to end training</em></p>

Unlike the previous case the feature extraction kernels weren't frozen. Initially the network was trained on the same dataset used in transfer learning. A target instance was picked and poison instances were generated using the iterative procedure. The model was retrained end to end on the poisoning instances + original train dataset.  Unlike the previous case where a single instance was used, here we generate 20 poison instances for a target instance.  Watermarking is used for every poison instance.
Our hyperparameters are set as follows:

$\beta=16, \lambda = 10^{-4}$, Opacity for Watermarking =0.2, Total iterations = 1000

Test accuracy of the classifier before poisoning = 92.4%

Average test accuracy of the classifier after poisoning = 87.5%

We obtain a success rate of 43% even with the watermarking. Shown below  is an example of a target instance and some of its poisoning instances. Again the poisoned examples have imperceptible changes from their base class. 

![e2e examples](images/Poisone2e.png)
<p align="center"><em>Poisoned images for attacks on end to end training</em></p>



 
Poisoning attacks on end to end training are much more difficult to execute then on a transfer learning task. The first reason is that poisoning visibly worsens the performance of the classsifier at test time and can be detected. [1] shows that unlike the transfer learning scenario where the final layer decision boundary rotates to accommodate the poison instance within the base region, the decision boundary in the end-to-end training scenario is unchanged after retraining on the poisoned dataset. The observation that can be made is during retraining the lower layers of the network adjust such that the poison instances are projected back to the base class in the higher level feature space. Quoting the paper directly: 

 "*The poison instance generation exploits imperfections in the feature extraction kernels in earlier layers such that the poison instance is placed alongside the target in feature space. When the network is retrained on this poison instance, because it is labeled as a base, those early-layer feature kernel imperfections are corrected and the poison instance is returned to the base class distribution.*" The key to a succesful attack in this scenario is to prevent the separation of the target instance and the poisoning examples during retraining. Watermarking achieves this goal to a certain exent. By blending the features of a target instance with its poisoned example, we bring the poisoned example extremely close to the target.

### Challenges 

The major challenged faced by us while implementing this attack was tuning the hyperparameter for generating posioned instance. One configuration doesn't fit all the base instances. A higher $\beta$ can make the poisoned instance resemble the base class but can seperate the target and poison instance in the feature space. On the other hand a hihger learning rate will give good feature collisions but the poisoned instance will look nothing like the base class and hence can be easily detected. Tuning should be done keeping this tradeoff in mind. 

To conclude the paper shows how flaws in neural networks can be exploited during train time and calls for attention to the important issue of data reliability

All code to implement the attacks can be found [here](https://github.com/SConsul/Adversarial_Attacks/tree/master/Poison_attacks)




 
## References

[1] [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks” by Shafahi et. al at NeurIPS 2018](https://papers.nips.cc/paper/7849-poison-frogs-targeted-clean-label-poisoning-attacks-on-neural-networks.pdf)


 



















