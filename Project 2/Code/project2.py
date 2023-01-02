import cv2
import numpy as np
from Minimum_failrate_Classifier import Minimum_failrate_Classifier
from Minimum_squared_error_Classifier import Minimum_squared_error_Classifier
from Nearest_neighbor_Classifier import Nearest_neighbor_Classifier
from PIL import Image
from numba import jit

def get_selected_image_index(image):
    # Select ROI
    r = cv2.selectROI(image, False, False)
    #return (start_x, start_y, width, hight)
    #To use it we need (start_x, start_y, start_x+width, start_y+hight)

    return r

def prepare_data_set(image_file, numbers_of_feature):
    image = cv2.imread(image_file)
    #Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    #Select feature from the image
    index_set = []
    for i in range(numbers_of_feature):
        index_set.append(get_selected_image_index(image))

    x_train = []
    y_train = []
    class_label = 0

    for index in index_set:
        x = np.copy(image_rgb[index[1]:index[1]+index[3],index[0]:index[0]+index[2]])
        x = x.reshape(x.shape[0]*x.shape[1],3)
        #flatten the sub image
        x_train +=x.tolist()
        #build the target label
        y_train += [class_label]*len(x)
        class_label += 1

    return np.array(x_train), np.array(y_train)

@jit(forceobj=True)
def pixel_processing(cl,image_rgb,lst_color):
    for i in range(image_rgb.shape[0]):
        for j in range(image_rgb.shape[1]):
            image_rgb[i][j] = lst_color[cl.predict(image_rgb[i][j])]
    return image_rgb

def segmentation(cl, image_file,lst_color):
    image = cv2.imread(image_file)
    #Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb = pixel_processing(cl,image_rgb,lst_color)
    cv2.imwrite("seg_image.png",cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

AQUA_RGB = [0,255,255]
YELLOW_RGB = [255,255,0]
BLUE_RGB = [0,0,255]


x_train, y_train = prepare_data_set("image/Bilde1.png", 3)
mfc = Minimum_failrate_Classifier()
mfc.fit(x_train,y_train)
segmentation(mfc, "image/ch.jpg",np.array([AQUA_RGB,YELLOW_RGB,BLUE_RGB]))
