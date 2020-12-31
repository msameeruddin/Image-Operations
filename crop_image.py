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

def crop_this(image_file, start_pos, length, width, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    image_shape = image_src.shape
    
    length = abs(length)
    width = abs(width)
    
    start_row = start_pos if start_pos >= 0 else 0
    start_column = start_row
    
    end_row = length + start_row
    end_row = end_row if end_row <= image_shape[0] else image_shape[0]
    
    end_column = width + start_column
    end_column = end_column if end_column <= image_shape[1] else image_shape[1]
    
    print("start row \t- ", start_row)
    print("end row \t- ", end_row)
    print("start column \t- ", start_column)
    print("end column \t- ", end_column)
    
    image_cropped = image_src[start_row:end_row, start_column:end_column]
    cmap_val = None if not gray_scale else 'gray'
    
    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Cropped")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_cropped, cmap=cmap_val)
        plt.show()
        return True
    return image_cropped

if __name__ == '__main__':
    crop_this(
        image_file="lena_original.png", 
        start_pos=199, 
        length=200, 
        width=200, 
        with_plot=True, 
    )