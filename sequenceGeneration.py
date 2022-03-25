# -*- coding: utf-8 -*-
"""
@author: Ivan Nikolov

Script for generating synthetic image parts on top of real backgrounds using the trained VAE

The script requires:
    -the trained model
    -cropped images as input - you can use the training images
    -bounding box positions and sizes for the input images
    -background images - the background images were created from the dataset Long-Term Thermal Dataset
https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset using a median filter on every 50 images to remove moving objects

The script contains the functions:
    - linearInterp_latentSpace - which takes two images, the VAE model, number of sequences of the latent space to be generated
between the two images and the input and output image size and generates an interpolated new image between the two inputs
    - makeRandomImage - which takes the model, a list of input sub-images, their positions and a background and generates
new images from the latent space of the VAE, interpolates their position between the positions of the two inputs and blends
them with the background image using Poisson blending. Each generated image can have a set or random number of synthetic people on it
    - load_backgrounds_from_folder - loads all the background images provided and selects a random background to use for the 
generation process.


The script can generate a set number of random images of randomly selecting backgrounds, input sub-images and generating 
new ones from the latent space of the VAE. It saves the images, as well as bounding boxes for the generated synthetic humans



"""
import numpy as np
from PIL import Image
import cv2
import random
import glob
import os


import torch
import torchvision
from torchvision import transforms

from model import VAE



# Function for extracting new images from the latent space between two input images in the VAE
def linearInterp_latentSpace(model, img_first,img_second, infer_size = 128,res_size = 74, num_gen = 50):
    
    # Resize the inputs and then trainsform the into tensors
    pre_process = transforms.Compose(
            [transforms.Resize((infer_size,infer_size)),
                transforms.ToTensor()])
    
    # Do the pre-processing for the two inputs
    img_first = pre_process(img_first)
    img_second = pre_process(img_second)
    
    # Set the model to evaluation
    model.eval()
    
    with torch.no_grad():
        
        # Stack the two images as a batch of size of 2
        data = torch.stack([img_first,img_second], dim=0)
        
        #  check if cuda is available
        if cuda:
            data = data.cuda() 
    
        # extract the latent space z
        z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
        
        # split the latent space and change its shape
        it = iter(z.split(1))
        z = zip(it, it)
        
        # Set the required number of samples to be generated from the latent space and create a linear space object from it
        # Generate the necessary llatent space selection
        zs = []
        numsample = num_gen
        for i,j in z:
            for factor in np.linspace(0,1,numsample):
                zs.append(i+(j-i)*factor)
        z = torch.cat(zs, 0)
        # decode the new latent space 
        recon = model.decode(z)
    

    # transform the reconstruction to a correct dimension and to numpy
    results_np = recon.cpu().permute(0, 2, 3, 1).detach().numpy()
    results_np_res = np.zeros([results_np.shape[0], res_size,res_size,results_np.shape[3]])
    # resize the reconstruction to the required output shape
    for i in range(0, results_np.shape[0]):
        
        results_np_res[i,:] =  cv2.resize(results_np[i], (res_size,res_size))
        
    
    return results_np_res

# Function for generating random images from the a list of images, their positons and backgrounds and the VAE
def makeRandomImage(model, numPeople, list_images, centers_array, image_background, num_gen = 50):
    
    # transform the chosen background images to uint8 and put it between 0-255
    image_background = image_background*255
    image_background = image_background.astype('uint8')
    
    labels = []
    # for the required number of synthetic people we do the generation
    for i in range(0, numPeople):
        # two random images are selected from the input sub-images of people
        img_first_ind, img_second_ind = np.random.choice(list_images.shape[0], 2, replace=False)
        
        imageFolder = r""
        #  first image
        img_name_first = list_images[img_first_ind]
        #  second image
        img_name_second = list_images[img_second_ind]
        
        # find their center positions
        center_first = centers_array[img_first_ind]
        center_second = centers_array[img_second_ind]
        
        x_coords = [center_first[0],center_second[0]]
        y_coords = [center_first[1],center_second[1]]
        
        
        # interpolate between the X and Y center coordinates of the two input images. We use this to choose where
        # the newly created synthetic sub image of a person would be put on
        x_interp = np.linspace(x_coords[0], x_coords[1], num_gen).astype(np.int)
        y_interp = np.linspace(y_coords[0], y_coords[1], num_gen).astype(np.int)
        
        # read the images
        img_first = Image.open(os.path.join(imageFolder, img_name_first)).convert('RGB')
        img_second = Image.open(os.path.join(imageFolder, img_name_second)).convert('RGB')
        
        # generate the synthetic person images between the two inputs from VAE
        results_np_res = linearInterp_latentSpace(model, img_first,img_second,num_gen = num_gen)
    
        #  select a random one from all the generated images 
        rand_ind = np.random.randint(0, results_np_res.shape[0])
        
        one_small = results_np_res[rand_ind,...]
        
        
        # get the new coordinates of the generated image
        y_coord = y_interp[rand_ind]
        x_coord = x_interp[rand_ind]
        
        
        
        #  transform the new synthetic image to uint and between 0-255
        one_small = one_small*255
        one_small = one_small.astype(np.uint8)
        
        # blur the image
        img_for_thresh = cv2.GaussianBlur(one_small[:,:,0],(7,7),0  ) 
        #  threshold the blurred image to remove any generated artifacts from the VAE
        _,threshold = cv2.threshold(img_for_thresh,10,255,cv2.THRESH_BINARY)
        
        #  get the contour of the threshold
        contours,_ = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        # get the bounding box of the contour
        x,y,w,h = cv2.boundingRect(cnt)
        
        #  Cut the small generated person sub-image. This is done to remove the black border used to make all inputs the same size
        one_small_cropped_scaled = one_small[y:y+h,x:x+w]
        
        
        
        
        try:

            #  make it into uint8
            one_small_cropped_scaled = one_small_cropped_scaled.astype('uint8')
           
            # generate a mask of the generated small image which is just white as the bounding box is quite tight around the object
            mask = 255 * np.ones(one_small_cropped_scaled.shape, one_small_cropped_scaled.dtype)
            

            #  check that the generated position of the new synthetic sub-image is in the bounds of the background
            if (one_small_cropped_scaled.shape[0]/2 > 2 and one_small_cropped_scaled.shape[1]/2 > 2 
                and y_coord + one_small_cropped_scaled.shape[0]/2 < image_background.shape[0]  
                and x_coord - one_small_cropped_scaled.shape[1]/2 > 0 
                and x_coord + one_small_cropped_scaled.shape[1]/2 < image_background.shape[1]):
               
                #  use Poisson blending to put the synthetic person sub-image on the newly generated coordinates. This removes some of the edges
                image_background = cv2.seamlessClone(one_small_cropped_scaled, image_background, mask, (x_coord,y_coord), cv2.NORMAL_CLONE)
            
                
                # save the labels 
                labels.append([0, x_coord/image_background.shape[1], y_coord/image_background.shape[0],
                               one_small_cropped_scaled.shape[1]/image_background.shape[1],
                               one_small_cropped_scaled.shape[0]/image_background.shape[0]])
                   

        except:
          print("An exception occurred")
          continue
        
        
        
        
    
    return image_background, labels

# Function to load all the background images
def load_backgrounds_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images




if __name__ == "__main__":

    # Load the best VAE model, the sub-images, their bounding boxes, the background images
    modelPath = "saved_model/best.pth"
    imageFolder = "data_preprocessing/bbox_images/train"
    bbox_file = "data_preprocessing/bbox_pos_sizes.txt"
    
    background_dir = r"background_images"
    
    
    output_folder = "generated_data"
    
    # Load the VAE model
    model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500)
    
    # check for cuda availability
    cuda = torch.cuda.is_available()
    
    if cuda:
        model.to("cuda")
        torch.cuda.empty_cache()
        torch.cuda.manual_seed(2) 
        
        
    savedStates = torch.load(modelPath)
    
    model.load_state_dict(savedStates.get("state_dict"))
    
    
    # make a list of all the usable input sub-images
    list_images = [f for f in glob.glob(os.path.join(imageFolder, "*.jpg"))]
    list_images = np.array(list_images)
    
    # load all the bounding box shapes and positions
    bbox_pos_size = np.loadtxt(bbox_file,ndmin=2)
    centers_array = bbox_pos_size[:,:2]
    
    
    # load all the backgrounds
    all_backgrounds = load_backgrounds_from_folder(background_dir)
        
    
    # set how many synthetic images do you want ot make, from which image number you want to start saving them and 
    # the size of the latent space to be interpolated
    num_images = 5
    
    start_number = 0
    num_lin_interp = 10
    
    for i in range(0, num_images):
        
        curr_num_pedestrians = random.randint(2,40)
        img_background = random.choice(all_backgrounds)/255
       
        output_image, output_labels = makeRandomImage(model, curr_num_pedestrians, list_images, centers_array, img_background,num_lin_interp) 
    
        cv2.imwrite(os.path.join(output_folder,"images",f"image_{str(i+start_number)}.jpg"), output_image)
        np.savetxt(os.path.join(output_folder,"labels",f"image_{str(i+start_number)}.txt"), np.array(output_labels), delimiter=' ', fmt='%1.3f')
        print(f"finished {i+1}/{num_images}")