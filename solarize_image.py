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

def solarize_this(image_file, thresh_val=128, with_plot=False, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    
    if not gray_scale:
        r_image, g_image, b_image = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
        r_sol = np.where((r_image < thresh_val), r_image, ~r_image)
        g_sol = np.where((g_image < thresh_val), g_image, ~g_image)
        b_sol = np.where((b_image < thresh_val), b_image, ~b_image)
        image_sol = np.dstack(tup=(r_sol, g_sol, b_sol))
    else:
        image_sol = np.where((image_src < thresh_val), image_src, ~image_src)
    
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Solarized")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_sol, cmap=cmap_val)
        plt.show()
        return True
    return image_sol

if __name__ == '__main__':
    solarize_this(
        image_file='lena_original.png', 
        with_plot=True, 
        gray_scale=True
    )