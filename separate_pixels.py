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

def separate_rgb(image_file, gray_scale=False):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    if not gray_scale:
        r_, g_, b_ = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
        return [r_, g_, b_]
    return [image_src]

def plot_rgb_seperated(image_file, with_plot=True, gray_scale=False):
    with_plot = True
    image_src = read_this(image_file=image_file, gray_scale=False)
    
    separates = separate_rgb(image_file=image_file, gray_scale=gray_scale)
    pixel_matrices = [image_src]
    pixel_matrices.extend(separates)
    
    cmap_vals = [None, 'Reds', 'Greens', 'Blues'] if not gray_scale else [None, 'gray']
    titles = ['Original', 'Red', 'Green', 'Blues'] if not gray_scale else ['Original', 'Grayscale']
    n_cols = 4 if not gray_scale else 2
    fig_size = (15, 10) if not gray_scale else (10, 20)
    
    if with_plot:
        fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=fig_size)
        for i, ax in zip(range(n_cols), axes):
            ax.axis("off")
            ax.set_title(titles[i])
            ax.imshow(pixel_matrices[i], cmap=cmap_vals[i])
        plt.show()
        return True
    return False

if __name__ == '__main__':
    plot_rgb_seperated(image_file='lena_original.png')