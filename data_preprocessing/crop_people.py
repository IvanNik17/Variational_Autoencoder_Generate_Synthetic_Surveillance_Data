# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 10:13:30 2022

@author: IvanTower
"""
import numpy as np
import cv2
import os
import glob




if __name__ == "__main__":
    image_dir = "images"
    bbox_dir = "bboxes"
    
    output_dir = "bbox_images/train"
    
    
    all_images = glob.glob(os.path.join(image_dir,"*.jpg"))
    all_bboxes = glob.glob(os.path.join(bbox_dir,"*.txt"))
    
    img_size = np.array([384,288])
    
    largest = np.array([-1,-1])
    for curr_bb_path in all_bboxes:
        curr_bboxes = np.loadtxt(curr_bb_path,delimiter=" ",ndmin=2)
        curr_max_width = curr_bboxes[:,3].max()
        curr_max_height = curr_bboxes[:,4].max()
        
        largest = [max(largest[0], curr_max_width), max(largest[1], curr_max_height)]
    
    largest = (largest * img_size).astype(int)
    
    all_bbox_imgs = []
    counter = 0
    for curr_im_path,curr_bb_path in zip(all_images,all_bboxes):
        
        curr_image = cv2.imread(curr_im_path,0)
        curr_bboxes = np.loadtxt(curr_bb_path,delimiter=" ",ndmin=2)
        
        for bbox in curr_bboxes:
            _,x,y,width,height = bbox
            
            x = int(x * img_size[0])
            width = int(width * img_size[0])
            y = int(y * img_size[1])
            height = int(height * img_size[1])
            
            cv2.rectangle(curr_image,(x-width//2,y-height//2),(x+width//2,y+height//2),(0,255,0),2)
            
            # image = cv2.putText(curr_image, f'{x}, {y}', [x-30,y], cv2.FONT_HERSHEY_SIMPLEX, 
            #        0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            img_small = curr_image[y-height//2:y+height//2,x-width//2:x+width//2]
            background_img = np.zeros([largest.max(), largest.max()]).astype(np.uint8)
            w_b, h_b = background_img.shape
            w_sm, h_sm = img_small.shape
            background_img[w_b//2-w_sm//2: w_b//2+w_sm//2, h_b//2-h_sm//2: h_b//2+h_sm//2] = img_small
            
            # cv2.imwrite(os.path.join(output_dir,f"image_{str(counter).zfill(5)}.jpg"),background_img)
            
            all_bbox_imgs.append([x,y,width,height])
            counter+=1
            
            # cv2.imshow("small", background_img)
            # key = cv2.waitKey(0)
            # if key == 27:
            #     break
            
            
        cv2.imshow("test", curr_image)
        
        
 
        key = cv2.waitKey(0)

        if key == 27:
            break
        
    all_bbox_imgs_arr = np.array(all_bbox_imgs)
    
    # np.savetxt('bbox_pos_sizes.txt', all_bbox_imgs_arr, delimiter=' ')