import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src

def resize_image(image_matrix, nh, nw):
    image_size = image_matrix.shape
    oh = image_size[0]
    ow = image_size[1]   
    
    re_image_matrix = np.array([
        np.array([image_matrix[(oh*h // nh)][(ow*w // nw)] for w in range(nw)]) 
        for h in range(nh)
    ])
    
    return re_image_matrix

def concat_images(image_set, how, with_plot=False):
    # dimension of each matrix in image_set
    shape_vals = [imat.shape for imat in image_set]
    
    # length of dimension of each matrix in image_set
    shape_lens = [len(ishp) for ishp in shape_vals]
    
    # if all the images in image_set are read in same mode
    channel_flag = True if len(set(shape_lens)) == 1 else False
    
    if channel_flag:
        ideal_shape = max(shape_vals)
        images_resized = [
            # function call to resize the image
            resize_image(image_matrix=imat, nh=ideal_shape[0], nw=ideal_shape[1]) 
            if imat.shape != ideal_shape else imat for imat in image_set
        ]
    else:
        return False
    
    images_resized = tuple(images_resized)
    
    if (how == 'vertical') or (how == 0):
        axis_val = 0
    elif (how == 'horizontal') or (how == 1):
        axis_val = 1
    else:
        axis_val = 1
    
    # numpy code to concatenate the image matrices
    # concatenation is done based on axis value
    concats = np.concatenate(images_resized, axis=axis_val)
    
    if with_plot:
        cmap_val = None if len(concats.shape) == 3 else 'gray'
        plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.imshow(concats, cmap=cmap_val)
        plt.show()
        return True
    return concats

if __name__ == '__main__':
    image1 = read_this(image_file='lena_original.png')
    image2 = read_this(image_file='pinktree.jpg')
    
    concat_images(
        image_set=[image1, image2, image1, image1, image2], 
        how='horizontal', 
        with_plot=True
    )