import cv2
import numpy as np
import json
from matplotlib import pyplot as plt

class ImageOperations(object):
    def __init__(self, image_file):
        self.image_file = image_file
        self.MAX_PIXEL = 255
        self.MIN_PIXEL = 0
        self.MID_PIXEL = self.MAX_PIXEL // 2
        self.color_db = self.read_colordb()
    
    def read_this(self, gray_scale=False):
        image_src = cv2.imread(self.image_file)
        if gray_scale:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        else:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        return image_src
    
    def read_colordb(self):
        with open(file='color_names_data.json', mode='r') as col_json:
            color_db = json.load(fp=col_json)
        return color_db
    
    def mirror_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_mirror = np.fliplr(image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_mirror, head_text='Mirrored', gray_scale=gray_scale)
            return None
        return image_mirror
    
    def flip_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_flip = np.flipud(image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_flip, head_text='Flipped', gray_scale=gray_scale)
            return None
        return image_flip
    
    def enhance_contrast(self, image_matrix):
        image_flattened = image_matrix.flatten()
        image_hist = np.zeros((self.MAX_PIXEL + 1))
        
        # frequency count of each pixel
        for pix in image_matrix:
            image_hist[pix] += 1
        
        # cummulative sum
        cum_sum = np.cumsum(image_hist)
        norm = (cum_sum - cum_sum.min()) * self.MAX_PIXEL
        # normalization of the pixel values
        n_ = cum_sum.max() - cum_sum.min()
        uniform_norm = norm / n_
        uniform_norm = uniform_norm.astype('int')
        
        # flat histogram
        image_eq = uniform_norm[image_flattened]
        # reshaping the flattened matrix to its original shape
        image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)
        return image_eq
    
    def equalize_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        if not gray_scale:
            r_image = image_src[:, :, 0]
            g_image = image_src[:, :, 1]
            b_image = image_src[:, :, 2]

            r_image_eq = self.enhance_contrast(image_matrix=r_image)
            g_image_eq = self.enhance_contrast(image_matrix=g_image)
            b_image_eq = self.enhance_contrast(image_matrix=b_image)

            image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        else:
            image_eq = self.enhance_contrast(image_matrix=image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_eq, head_text='Equalized', gray_scale=gray_scale)
            return None
        return image_eq
    
    def convert_binary(self, image_matrix, thresh_val, colors=None):
        color_1 = self.MAX_PIXEL
        color_2 = self.MIN_PIXEL

        if len(image_matrix.shape) == 3:
            if colors and len(colors) == 2:
                colors_list = list(self.color_db.keys())
                colors = [c.lower().strip() for c in colors]
                
                color_1 = np.array([self.color_db[colors[0]][i] for i in 'rgb'])
                color_2 = np.array([self.color_db[colors[1]][i] for i in 'rgb'])

        initial_conv = np.where((image_matrix <= thresh_val), image_matrix, color_1)
        final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
        return final_conv

    def binarize_this(self, with_plot=False, gray_scale=False, colors=None):
        image_src = self.read_this(gray_scale=gray_scale)
        image_b = self.convert_binary(image_matrix=image_src, thresh_val=self.MID_PIXEL, colors=colors)

        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_b, head_text='Binarized', gray_scale=gray_scale)
            return None
        return image_b
    
    def invert_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_i = ~ image_src
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_i, head_text='Inverted', gray_scale=gray_scale)
            return None
        return image_i
    
    def crop_this(self, start_pos, length, width, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
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
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_cropped, head_text='Cropped', gray_scale=gray_scale)
            return None
        return image_cropped
    
    def draw_border(self, bt=5, with_plot=False, gray_scale=False, color_name=0):
        image_src = self.read_this(gray_scale=gray_scale)    
        if gray_scale:
            color_name = 0
            image_bord = np.pad(array=image_src, pad_width=bt, mode='constant', constant_values=color_name)
        else:
            color_name = str(color_name).strip().lower()            
            colors_list = list(self.color_db.keys())
            
            if color_name not in colors_list:
                r_cons, g_cons, b_cons = [0, 0, 0]
            else:
                r_cons, g_cons, b_cons = [self.color_db[color_name][i] for i in 'rgb']

            r_, g_, b_ = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
            rb = np.pad(array=r_, pad_width=bt, mode='constant', constant_values=r_cons)
            gb = np.pad(array=g_, pad_width=bt, mode='constant', constant_values=g_cons)
            bb = np.pad(array=b_, pad_width=bt, mode='constant', constant_values=b_cons)

            image_bord = np.dstack(tup=(rb, gb, bb))

        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_bord, head_text='Bordered', gray_scale=gray_scale)
            return None
        return image_bord
    
    def draw_rectangle(self, start_pos, length, width, thickness=3, with_plot=False, gray_scale=False, color_name=0):
        image_src = self.read_this(gray_scale=gray_scale)
        image_main = self.read_this(gray_scale=gray_scale)
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
                max_width, max_height = gso_image.shape
                inner_image_rect = np.zeros(shape=(max_width, max_height))
        else:
            color_name = str(color_name).strip().lower()
            colors_list = list(self.color_db.keys())

            if color_name not in colors_list:
                r_cons, g_cons, b_cons = (0, 0, 0)
            else:
                r_cons, g_cons, b_cons = [self.color_db[color_name][i] for i in 'rgb']

            r_inner_image, g_inner_image, b_inner_image = gsi_image[:, :, 0], gsi_image[:, :, 1], gsi_image[:, :, 2]

            if thickness != -1:
                r_inner_rect = np.pad(array=r_inner_image, pad_width=thickness, mode='constant', constant_values=r_cons)
                g_inner_rect = np.pad(array=g_inner_image, pad_width=thickness, mode='constant', constant_values=g_cons)
                b_inner_rect = np.pad(array=b_inner_image, pad_width=thickness, mode='constant', constant_values=b_cons)
                inner_image_rect = np.dstack(tup=(r_inner_rect, g_inner_rect, b_inner_rect))
            else:
                max_width, max_height, _ = gso_image.shape
                r_out_rect = np.full(shape=(max_width, max_height), fill_value=r_cons)
                g_out_rect = np.full(shape=(max_width, max_height), fill_value=g_cons)
                b_out_rect = np.full(shape=(max_width, max_height), fill_value=b_cons)
                inner_image_rect = np.dstack(tup=(r_out_rect, g_out_rect, b_out_rect))

        image_src[start_row_grab:end_row_grab, start_column_grab:end_column_grab] = inner_image_rect
        image_rect = image_src    

        if with_plot:
            self.plot_it(orig_matrix=image_main, trans_matrix=image_rect, head_text='Rectangle', gray_scale=gray_scale)
            return None
        return image_rect
    
    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        cmap_val = None if not gray_scale else 'gray'
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text(head_text)
        
        ax1.imshow(orig_matrix, cmap=cmap_val)
        ax2.imshow(trans_matrix, cmap=cmap_val)
        plt.show()
        return True


if __name__ == '__main__':
    imo = ImageOperations(image_file='lena_original.png')
    imo.mirror_this(with_plot=True)