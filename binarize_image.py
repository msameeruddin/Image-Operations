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

def convert_binary(image_matrix, thresh_val, colors=None):
    color_1 = 255
    color_2 = 0
    cmap_val = None
    
    if len(image_matrix.shape) == 3:
        if colors and len(colors) == 2:            
            with open(file='color_names_data.json', mode='r') as col_json:
                color_db = json.load(fp=col_json)
            
            colors_list = list(color_db.keys())
            colors = [c.lower().strip() for c in colors]
            color_1 = np.array([color_db[colors[0]][i] for i in 'rgb'])
            color_2 = np.array([color_db[colors[1]][i] for i in 'rgb'])
    else:
        cmap_val = 'gray'
    
    initial_conv = np.where((image_matrix <= thresh_val), image_matrix, color_1)
    final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
    
    return cmap_val, final_conv

def binarize_this(image_file, thresh_val=127, with_plot=False, gray_scale=False, colors=None):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    cmap_val, image_b = convert_binary(image_matrix=image_src, thresh_val=thresh_val, colors=colors)
    
    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Binarized")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        plt.show()
        return True
    return image_b

if __name__ == '__main__':
    binarize_this(image_file='lena_original.png', with_plot=True)