from os import listdir,walk
import cv2,pickle,numpy as np,matplotlib.pyplot as plt,matplotlib.image as mpimg,os
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report,precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from PIL import Image
from caps import build_model

from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

def get_subsubfolder(folder_name):
	folder = []
	for i in listdir(folder_name):
		folder.append(folder_name+"/"+i)
	return(folder)

def get_file(folder):
    filename = []
    for i in listdir(folder):
        filename.append(folder+"/"+i)
    return(filename)

def get_1set_files(folder):
    folder = get_subsubfolder(folder)
    filename = [get_file(i) for i in folder]
    res = []
    for i in filename:
        res = res+i
    return res

def open_img(filename,thresh,nkernel,th1,th2):
    img = cv2.imread(filename,0)
    #img = img[:,:,1]
    img = cv2.resize(img,(28,28))
    kernel=np.ones((nkernel,nkernel),np.uint8)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)[1]
    img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.bitwise_not(img)
    img = cv2.Canny(img,th1,th2)
    img = img.flatten()
    img = img.astype('float32')
    return img/255

def open_images(filenames,thresh,nkernel,th1,th2):
    return np.asarray([open_img(i,thresh,nkernel,th1,th2) for i in filenames])

def get_label(filename):
    return[str.split(i,"/")[-2] for i in filename]

def categorize(label):
    hsl=[]
    lab = np.unique(label)
    for i in range(len(label)):
        for j in range(len(lab)):
            if label[i] == lab[j]:
                hsl.append(j)
    return np.asarray(hsl)

def prediksi(X):
    capsmod = build_model(0)
    capsmod.load_weights(os.path.dirname(os.path.realpath(__file__))+"/db/"+'model.h5')
    aa = X.reshape(-1,28,28,1)
    return capsmod.predict(aa,batch_size=50)