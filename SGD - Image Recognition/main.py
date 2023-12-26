import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
pp = pprint.PrettyPrinter(indent=4)
import joblib
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from skimage.color import rgb2gray
 
def resize_all(src, pklname, include, width=150, height=None):     
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)
    return data

def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)
 
 
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

def main_function():
    data_path = fr'{os.getenv("HOME")}/Documents/DEV/IA/Image'
    base_name = 'animal_faces'
    width = 80
    include = {'ChickenHead', 'BearHead', 'ElephantHead', 'EagleHead', 'DeerHead', 'MonkeyHead', 'PandaHead'}

    data = resize_all(src=data_path, pklname=base_name, width=width, include=include)

    labels = np.unique(data['label'])

    fig, axes = plt.subplots(1, len(labels))
    fig.set_size_inches(15,4)
    fig.tight_layout()


    for ax, label in zip(axes, labels):
        idx = data['label'].index(label)
         
        ax.imshow(data['data'][idx])
        ax.axis('off')
        ax.set_title(label)


    X = np.array(data['data'])
    y = np.array(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        shuffle=True,
        random_state=42,
    )

    # plt.suptitle('relative amount of photos per type')
    # plot_bar(y_train, loc='left')
    # plot_bar(y_test, loc='right')
    # plt.legend([
    #     'train ({0} photos)'.format(len(y_train)), 
    #     'test ({0} photos)'.format(len(y_test))
    # ]);

    # create an instance of each transformer
    grayify = RGB2GrayTransformer()
    hogify = HogTransformer(pixels_per_cell=(14, 14), cells_per_block=(2,2), orientations=9, block_norm='L2-Hys')
    scalify = StandardScaler()
     
    # call fit_transform on each transform converting X_train step by step
    X_train_gray = grayify.fit_transform(X_train)
    X_train_hog = hogify.fit_transform(X_train_gray)
    X_train_prepared = scalify.fit_transform(X_train_hog)
     
    sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    sgd_clf.fit(X_train_prepared, y_train)

    X_test_gray = grayify.transform(X_test)
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify.transform(X_test_hog)

    y_pred = sgd_clf.predict(X_test_prepared)
    # print(np.array(y_pred == y_test)[:25])
    # print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

    return sgd_clf, grayify, hogify, scalify


############################ TESTS #############################





