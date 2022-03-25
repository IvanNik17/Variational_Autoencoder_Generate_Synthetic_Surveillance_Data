# -*- coding: utf-8 -*-
"""

@author: Ivan Nikolov

Script for testing the trained VAE model. It contains functions for visualizing single and multiple images
Currently it uses the training images, but it can be changed to testing images
It also visualizes the input, reconstructed and the difference between the two

"""
import torch

from model import VAE

import cv2

import matplotlib.pyplot as plt

import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader




def show_image(img,name):
    img = img.permute(1, 2, 0).cpu()
    img = img.numpy()
    name = name + str(img.shape)
    cv2.imshow(name,img)
    
    
def show_images_grid(images):
    images = torchvision.utils.make_grid(images)
    show_image_plt(images)

def show_image_plt(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    
    
def rand_faces(model, num=5, cuda = True):

    model.eval()
    z = torch.randn(num*num, model.latent_variable_size)
    if cuda: 
        z = z.cuda()
    
    recon = model.decode(z)
    
    show_images_grid(recon.cpu())
    
    
dataset_path = "data_preprocessing/bbox_images"


modelPath = "saved_model/best.pth"

batch_size = 1    


model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500)
    
cuda = torch.cuda.is_available()

if cuda:
    model.to("cuda")
    torch.cuda.empty_cache()
    torch.cuda.manual_seed(2) 
    
    
savedStates = torch.load(modelPath)

model.load_state_dict(savedStates.get("state_dict"))

model.eval()



pre_process = transforms.Compose(
        [transforms.Resize((128,128)),
            transforms.ToTensor()])
    
test_data = torchvision.datasets.ImageFolder(dataset_path, transform=pre_process)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,  num_workers=0)


    
  
with torch.no_grad():
    for idx, (images, _) in enumerate(testloader):
        
        
        if cuda:
            test_img = images.cuda() 
        
        recon_out, mu, logvar = model(test_img)
        
        out_norm = recon_out[0]
        # show_image(out_norm,"rec")
        
        cv2.imshow('recon', recon_out.cpu().numpy()[0,0,:,:])
        cv2.imshow('input', images.cpu().numpy()[0,0,:,:])
        
        cv2.imshow('diff', images.cpu().numpy()[0,0,:,:] - recon_out.cpu().numpy()[0,0,:,:])
        
        key = cv2.waitKey(0)
    
        if key == 27:
            
            break
