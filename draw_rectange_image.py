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

def draw_rectangle(image_file, start_pos, length, width, thickness=3, with_plot=False, gray_scale=False, color_name=0):
    image_src = read_this(image_file=image_file, gray_scale=gray_scale)
    image_main = read_this(image_file=image_file, gray_scale=gray_scale)
    image_shape = image_src.shape
    
    length = abs(length)
    width = abs(width)
    thickness = -1 if thickness < 0 else thickness
    
    start_row = start_pos if start_pos >= 0 else 0
    start_column = start_row
    end_row = length + start_row
    end_row = end_row if end_row <= image_shape[0] else image_shape[0]
    end_column = width + start_column
    end_column = end_column if end_column <= image_shape[1] else image_shape[1]
    
    start_row_grab = start_row - thickness
    end_row_grab = end_row + thickness
    start_column_grab = start_row_grab
    end_column_grab = end_column + thickness
    
    gso_image = image_src[start_row_grab:end_row_grab, start_column_grab:end_column_grab]
    gsi_image = image_src[start_row:end_row, start_column:end_column]
    
    if gray_scale:
        color_name = 0
        if thickness != -1:
            inner_image_rect = np.pad(array=gsi_image, pad_width=thickness, mode='constant', constant_values=color_name)
        else:
            max_height, max_width = gso_image.shape
            inner_image_rect = np.zeros(shape=(max_height, max_width))
    else:
        with open(file='color_names_data.json', mode='r') as col_json:
            color_db = json.load(fp=col_json)
        
        color_name = str(color_name).strip().lower()
        colors_list = list(color_db.keys())
        
        if color_name not in colors_list:
            r_cons, g_cons, b_cons = (0, 0, 0)
        else:
            r_cons, g_cons, b_cons = [color_db[color_name][i] for i in 'rgb']
        
        r_inner_image, g_inner_image, b_inner_image = gsi_image[:, :, 0], gsi_image[:, :, 1], gsi_image[:, :, 2]
        
        if thickness != -1:
            r_inner_rect = np.pad(array=r_inner_image, pad_width=thickness, mode='constant', constant_values=r_cons)
            g_inner_rect = np.pad(array=g_inner_image, pad_width=thickness, mode='constant', constant_values=g_cons)
            b_inner_rect = np.pad(array=b_inner_image, pad_width=thickness, mode='constant', constant_values=b_cons)
            inner_image_rect = np.dstack(tup=(r_inner_rect, g_inner_rect, b_inner_rect))
        else:
            max_height, max_width, _ = gso_image.shape
            r_out_rect = np.full(shape=(max_height, max_width), fill_value=r_cons)
            g_out_rect = np.full(shape=(max_height, max_width), fill_value=g_cons)
            b_out_rect = np.full(shape=(max_height, max_width), fill_value=b_cons)
            inner_image_rect = np.dstack(tup=(r_out_rect, g_out_rect, b_out_rect))
    
    image_src[start_row_grab:end_row_grab, start_column_grab:end_column_grab] = inner_image_rect
    image_rect = image_src    
    
    if with_plot:
        cmap_val = None if not gray_scale else 'gray'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text("Rectangle")
        
        ax1.imshow(image_main, cmap=cmap_val)
        ax2.imshow(image_rect, cmap=cmap_val)
        plt.show()
        return True
    return image_rect

if __name__ == '__main__':
    draw_rectangle(
        image_file='lena_original.png', 
        start_pos=199, 
        length=200, 
        width=200, 
        thickness=3, 
        with_plot=True, 
        color_name='red'
    )