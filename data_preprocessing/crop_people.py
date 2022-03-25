# -*- coding: utf-8 -*-
"""

@author: Ivan Nikolov

Function for generating training data for the variational autoencoder. It takes images and bounding box annotations and
cuts them into individual small images. In the current implementation the images are taken from the Long-Term Thermal Dataset
https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset
The bounding box files are in the format - Object_Type,X,Y,Width,Height. The x,y,width and height need to be scaled using
the image size of 384x288. The Object_type is only represented as 0, as only pedestrians are annotated.

The small images are saved in the bbox_images/train folder, while the scaled size and positions are saved in a single text file

The small images are scaled to the same size by adding black border around them. The size is based on the size the biggest annotated image
This is done for easier feeding to the variational autoencoder


"""
import numpy as np
import cv2
import os
import glob




if __name__ == "__main__":
    image_dir = "images"
    bbox_dir = "bboxes"
    
    output_dir = "bbox_images/train"
    
    # Choose if you want to visualize the small cut sub images and the large image with bounding boxes drawn over the annotations
    visualize_small = False
    visualize_large = True
    
    # Load all the images and annotations
    all_images = glob.glob(os.path.join(image_dir,"*.jpg"))
    all_bboxes = glob.glob(os.path.join(bbox_dir,"*.txt"))
    
    # size of the images
    img_size = np.array([384,288])
    
    # Go through the annotations and find the one with the largest height and width and save them
    # This is going to be used for creating a black boarder around the saved images so they are of same size
    # This is necessary as some annotations are only 20-25 pixels while others are much larger
    largest = np.array([-1,-1])
    for curr_bb_path in all_bboxes:
        curr_bboxes = np.loadtxt(curr_bb_path,delimiter=" ",ndmin=2)
        curr_max_width = curr_bboxes[:,3].max()
        curr_max_height = curr_bboxes[:,4].max()
        
        largest = [max(largest[0], curr_max_width), max(largest[1], curr_max_height)]
    
    largest = (largest * img_size).astype(int)
    
    
    # Go through the images and bounding box files
    all_bbox_imgs = []
    counter = 0
    for curr_im_path,curr_bb_path in zip(all_images,all_bboxes):
        
        # read and image and bounding box file
        curr_image = cv2.imread(curr_im_path,0)
        curr_bboxes = np.loadtxt(curr_bb_path,delimiter=" ",ndmin=2)
        
        # copy the image for visualization purposes so we can draw rectangles on it without worrying that they will be saved
        curr_image_visual = curr_image.copy()
        # Get all the lines of the bounding box files and scale the values
        for bbox in curr_bboxes:
            _,x,y,width,height = bbox
            
            x = int(x * img_size[0])
            width = int(width * img_size[0])
            y = int(y * img_size[1])
            height = int(height * img_size[1])
            
            # Draw a rectangle around bounding box for visualization
            cv2.rectangle(curr_image_visual,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
            
            # get the ROI of the bounding box as an image
            img_small = curr_image[y-height//2:y+height//2,x-width//2:x+width//2]
            # Create the black background of the image with a square shape based on the largest annotated bounding box
            background_img = np.zeros([largest.max(), largest.max()]).astype(np.uint8)
            
            w_b, h_b = background_img.shape
            w_sm, h_sm = img_small.shape
            background_img[w_b//2-w_sm//2: w_b//2+w_sm//2, h_b//2-h_sm//2: h_b//2+h_sm//2] = img_small
            
            # Save the ROI small image
            cv2.imwrite(os.path.join(output_dir,f"image_{str(counter).zfill(5)}.jpg"),background_img)
            
            # append the bounding box coordinates
            all_bbox_imgs.append([x,y,width,height])
            counter+=1
            
            # visualize the small sub images and wait for any key
            if visualize_small:
                cv2.imshow("small", background_img)
                key = cv2.waitKey(0)
                if key == 27:
                    break
            
        # Visualize the large image with drawn rectangles and wait for any key or press ESC to exit
        if visualize_large:
            cv2.imshow("large", curr_image_visual)
            key = cv2.waitKey(0)
            if key == 27:
                break
    # Save the bounding box files 
    all_bbox_imgs_arr = np.array(all_bbox_imgs)
    
    np.savetxt('bbox_pos_sizes.txt', all_bbox_imgs_arr, delimiter=' ')