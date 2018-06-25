from keras.preprocessing import image
import labels
import cv2 as cv
from lib.bilinearUpsampling import BilinearUpSampling2D
import sys
import os
import re
import matplotlib.pyplot as plt

def plot_results(y_true, y_pred, cfMatrix=False, score=[],  history=None):    
    
    if not len(score):
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.reshape(y_pred, (y_pred.shape[0]*img_rows*img_cols))
    y_true = np.reshape(y_true, (y_true.shape[0]*img_rows*img_cols))

    
    if not len(history):
        # Learning Curve Plots
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.xlim(0, epochs)
        plt.xticks(np.arange(0, epochs+1, 5))
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlim(0, epochs)
        plt.xticks(np.arange(0, epochs+1, 5))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    report = classification_report(y_true, y_pred)
    print(report)

    meanIoU = jaccard_similarity_score(y_true, y_pred)  

    
    print('----- Mean IoU ----- ')
    print('------ {} -----------'.format(meanIoU))
    
    # Remove the last label
    if cfMatrix:
        cfMatrix = confusion_matrix(y_true, y_pred)
        cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=0)
        cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=1)   
        plt.figure()
        plot_confusion_matrix(cfMatrix, labels.listLabels)
        plt.show()


def load_image(imagePath, patchSize, show=False, scale=True, mean=None):
    """loads the image from the path and scales it properly

    Input:
        imagePath: The path of the image
        patchSize: The square size of the image
        scale: Scale to the right space
        mean: the mean frame from the training set 
    """
    img = image.load_img(imagePath)
    img = image.img_to_array(img)
    if scale:
        img = cv.resize(img, dsize=(img.shape[0]/2,img.shape[1]/4), interpolation=cv.INTER_CUBIC)
    img = np.expand_dims(img, axis=0)
    if not mean:
        #np.load(asdas) Load the mean frame
        img -= mean
    img /= 255.
   
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.show()
    img = np.reshape(img, ((img.shape[1]*img.shape[2])/(patchSize*patchSize), patchSize, patchSize, channels))
    return img 

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    
    Arguments 
    ==========
            Input: 
                cm: confusion matrix [Num_classes, Num_classes]

                classes: A list with the names of the labels

                normalize: Normalize the values 
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def computeGlobalIou(cfArray):
    """Takes the confussion matrix in order to extract the results
    """
    TP = np.diag(cfArray[:-1,:-1])
    FP = cfArray[:-1,:-1].sum(axis=0) - TP
    FN = cfArray[:-1,:-1].sum(axis=1) - TP

    overAll = np.true_divide(TP.sum(),(TP.sum()+FP.sum()+FN.sum()))
    overAll = overAll[~np.isnan(overAll)]

    return overAll

# Predictions to image
def labels2Image(y):
    print(y.shape)
    image_y = np.empty((1024, 2048, channels), dtype='uint8')
    y = np.reshape(y, (1024, 2048))

    for index, value in np.ndenumerate(y):
        image_y[index[0], index[1]] = labels.trainId2Color[value]
    # Change color space
    image_y  = cv.cvtColor(image_y, cv.COLOR_RGB2BGR)
    return image_y 

def resizeImage(image_y):
    if image_y.ndim == 1:
        image_y = np.reshape(image_y, (img_rows, img_cols))
    return cv.resize(image_y, dsize=(2048,1024), interpolation=cv.INTER_NEAREST)

def saveImage(image_y, filepath):
    filename = re.findall('\w+_*.png', imagePath)[0]
    cv.imwrite('./test_images/'+filename, image_y)

def medianFilter(image, ksize=15):
    return cv.medianBlur(image, ksize=ksize)


# Get the lists with images and ground truths
def getFileLists(testPath, gtTestPath):
    imageList   = []
    gtImageList = []

    for subdir, dirs, files in os.walk(testPath):
        for file in files:
            #print dirs
            imageList.append(os.path.join(subdir, file))
            
    for subdir, dirs,files in os.walk(gtTestPath):
            for file  in files:
                if re.findall( '\w*_labelTrainIds.png',file):
                    gtImageList.append(os.path.join(subdir, file))
    # Sort images
    imageList   = sorted(imageList)
    gtImageList = sorted(gtImageList)

    return imageList, gtImageList


def load_model(modelPath, weights=None):
    # load json and create model
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})
    if os.path.exists(weights):
        model.load_weights(weightsPath)
        print('Weights loaded successfully')       
    return model

if __name__ == '__main__':
    print('This module is not callable.')

    
