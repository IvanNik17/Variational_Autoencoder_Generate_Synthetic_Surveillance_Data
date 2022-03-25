# -*- coding: utf-8 -*-
"""
@author: Ivan Nikolov

Code based on the work from https://github.com/bhpfelix/Variational-Autoencoder-PyTorch

Code for training the Variational Autoencoder (VAE). Currently the code is simplified to only use training data,
but no validation. This can be easily added by separating the training data in train and validation, loading the validation data separately,
checking the performance of at specific training epochs and stopping as neccesary

The loss fucntion is set to a combination of Binary Cross Entropy BCE and Kullback–Leibler divergence KLD
The images are rescaled to 128x128 resolution
"""


import torch
import torch.nn.functional as F
from model import VAE


import os
from time import time

import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader


# Function for saving checkpoints of the model state
def save_checkpoint(state, output_dir, filename):
    torch.save(state, os.path.join(output_dir,filename))

# Loss function used for variational autoencoders 
def loss_function(recon_x, x, mu, logvar):
    
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

# Training function uses Adam optimizer a learning rate of 1e-4. 
def trainVAE(model,train_dataset_dir, model_output_dir,start_epoch, num_epochs,batch_size,learning_rate = 1e-4, cuda = True):
    
    # Load model
    model_output_dir = os.path.normpath(model_output_dir)
    
    # Setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    # Set the model to training
    model.train()
    
    # Preprocessing step of resizing the images and transforming them to tensors
    pre_process = transforms.Compose(
        [transforms.Resize((128,128)),
            transforms.ToTensor()])
    
    
    # Load Training data images, using shuffling
    train_data = torchvision.datasets.ImageFolder(train_dataset_dir, transform=pre_process)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=0)
    
    best_loss = 2.0e50
    
    loss_all_epoch = []
    #  Run training epochs to the prerequisity number of times
    for epoch in range(start_epoch, start_epoch+ num_epochs):
        
        
        print(f'\n<----- START EPOCH {epoch} ------->\n')

        start_time = time()
        
        train_loss = 0
        
        # batch counter for outputing training information 
        batch = 1
        for idx, (images, _) in enumerate(trainloader):
            
            optimizer.zero_grad()
            #  check for cuda
            if cuda:
                images = images.cuda()
            
            # get output of the encoder latent space, mean and variance
            recon_batch, mu, logvar = model(images)
            #  calculate the loss
            loss = loss_function(recon_batch, images, mu, logvar)
            #  propagate and step
            loss.backward()
            optimizer.step()
            
            # calculate current loss
            train_loss += loss.item()
            
            # print epoch and loss information
            if (batch-1) % 10 == 0:
                
                print(f'Epoch {epoch} Batch {batch} Loss {loss.item()}')
                
            batch+=1
            
        # append the loss
        avg_loss_epoch = train_loss/batch
        
        loss_all_epoch.append(avg_loss_epoch)
        
        
        
        # Save current model
        save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),},model_output_dir,'current.pth')
        print('Current model saved')    
            
        # save best model 
        if avg_loss_epoch < best_loss:
            best_loss = avg_loss_epoch
            
            save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),},model_output_dir,'best.pth')
         
        print(f'\n<----- END EPOCH {epoch} Time elapsed: {time()-start_time}------->\n')
    
    return [loss_all_epoch,best_loss]
            


if __name__ == '__main__':
    
    dataset_path = "data_preprocessing/bbox_images"
    
    output_dir = "saved_model"
    
    # set number of epochs and batch size
    num_epochs = 100
    batch_size = 32
    
    # set if you want to continue training of the model, in case of error or stopping prematurely
    continueTrain = False
    
    start_epoch = 0
    
    # create model with number of color channels, input size and size of latent space
    model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500)
    
    #  if continueing the training the load the previous best model information and setup the start epoch
    if continueTrain:
        modelPath = os.path.join(output_dir, "best.pth")
        
        savedStates = torch.load(modelPath)
    
        model.load_state_dict(savedStates.get("state_dict"))
        start_epoch = int(savedStates.get("epoch"))
    
    
    # check for cuda and setup as needed. Empty cache in case of old data still being inside
    cuda = torch.cuda.is_available()
    
    if cuda:
        model.to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(2)   
        
    # train 
    all_losses,best_loss = trainVAE(model,dataset_path,output_dir,start_epoch,num_epochs,batch_size) 