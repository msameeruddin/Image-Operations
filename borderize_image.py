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

def draw_border(image_file, bt=5, with_plot=False, gray_scale=False, color_name=0):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)    
    if gray_scale:
        color_name = 0
        image_b = np.pad(array=image_src, pad_width=bt, mode='constant', constant_values=color_name)
        cmap_val = 'gray'
    else:
        with open(file='color_names_data.json', mode='r') as col_json:
            color_db = json.load(fp=col_json)
        
        color_name = str(color_name).strip().lower()
        colors_list = list(color_db.keys())
        
        if color_name not in colors_list:
            r_cons, g_cons, b_cons = [0, 0, 0]
        else:
            r_cons, g_cons, b_cons = [color_db[color_name][i] for i in 'rgb']
        
        r_, g_, b_ = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
        rb = np.pad(array=r_, pad_width=bt, mode='constant', constant_values=r_cons)
        gb = np.pad(array=g_, pad_width=bt, mode='constant', constant_values=g_cons)
        bb = np.pad(array=b_, pad_width=bt, mode='constant', constant_values=b_cons)
        
        image_b = np.dstack(tup=(rb, gb, bb))
        cmap_val = None
    
    if with_plot:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Bordered")
        
        ax1.imshow(image_src, cmap=cmap_val)
        ax2.imshow(image_b, cmap=cmap_val)
        plt.show()
        return True
    return image_b

if __name__ == '__main__':
    draw_border(image_file='lena_original.png', with_plot=True, color_name="yellow")