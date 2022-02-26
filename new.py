import cv2
import numpy as np, pandas as pd, os
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
import b2
from caps import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import matplotlib.backends.tkagg as tkagg
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

class caps_run:
    def __init__(self,lr,batch,epoch,step):
        data = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+"/db/"+'data_img.csv')
        self.dt = data.drop('label',axis=1).values
        self.y = data['label'].values
        self.yy = b2.categorize(self.y)
        self.labels = unique_labels(self.y)
        X_train, X_test, y_train, self.y_tes = train_test_split(self.dt, self.yy, test_size=0.2)
        X_train = X_train.reshape(-1,28,28,1)
        X_test = X_test.reshape(-1,28,28,1)
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(self.y_tes)
        print(X_train.shape)
        print(X_test.shape)
        model = build_model(lr)
        aug = ImageDataGenerator(rotation_range=50, zoom_range=0.15,
            width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15,
            horizontal_flip=True, fill_mode="nearest")
        if step<2:
            step = len(X_train)/batch
        model.fit_generator(aug.flow(X_train, y_train, batch_size=batch),
                            validation_data=aug.flow(X_test, y_test,batch_size=batch),validation_steps=len(X_test)/batch, 
                            steps_per_epoch=step,epochs=epoch)
        model.save(os.path.dirname(os.path.realpath(__file__))+"/db/"+'model.h5')
        self.y_predpr = model.predict_generator(aug.flow(X_test,y_test,batch_size=batch),steps=len(y_test)/batch)
        self.y_pred = np.argmax(self.y_predpr,axis=1)
        self.score = accuracy_score(self.y_tes, self.y_pred)
        print(self.score)
        # self.predictions = np.array([labels[i] for i in self.y_pred])
        # self.truths = np.array([labels[i] for i in self.y_test])
        self.confmat = confusion_matrix(self.y_tes, self.y_pred)
        self.report = classification_report(self.y_tes, self.y_pred)

def plot_confusion_matrix(cm, classes,canvas,normalize=False, title=None,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        # else:
        #     title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(1,1,figsize=(2.8,2.8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    aa = FigureCanvasTkAgg(fig, canvas)
    aa.get_tk_widget().grid(row=0, column=0)


class get_image:
    def __init__(self,image_file,threshold,nkernel,th1,th2):
        img = Image.open(image_file)
        im1 = img.resize((200,160)) 
        im1 = im1.convert('RGB')
        self.img = im1
        im1 = im1.convert('LA')
        self.imgray = im1
        im1 = np.array(im1)
        kernel=np.ones((nkernel,nkernel),np.uint8)
        im1 = cv2.morphologyEx(im1, cv2.MORPH_CLOSE, kernel)
        self.immorph = Image.fromarray(im1)
        im1 = cv2.threshold(im1,threshold,255,cv2.THRESH_BINARY)[1]
        self.imbin = Image.fromarray(im1)
        im1= cv2.erode(im1,kernel,iterations = 1)
        self.imero = Image.fromarray(im1)
        im1 = cv2.bitwise_not(im1)
        self.iminv = Image.fromarray(im1)
        im1 = cv2.Canny(im1,th1,th2)
        self.imedge = Image.fromarray(im1)

def place_image(image,canvas,x,y,hg,wt):
    filename = ImageTk.PhotoImage(image)
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(0,0,anchor="nw",image=filename)
    canvas.pack()
    canvas.place(relx=x, rely=y, relheight=hg,relwidth=wt)