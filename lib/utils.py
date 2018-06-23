
def plot_results(history, y_true, y_pred, score, cfMatrix=False):    
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.reshape(y_pred, (y_pred.shape[0]*img_rows*img_cols))
    y_true = np.reshape(y_true, (y_true.shape[0]*img_rows*img_cols))

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

    # Compute mean IoU
    #meanAcc, labelIou, meanIoUfromcf = computeMeanIou(cfMatrix)
    
    print('----- Mean IoU ----- ')
    print('------ %s -----------'%(str(meanIoU)))
    
    # Remove the last label
    if cfMatrix:
        cfMatrix = confusion_matrix(y_true, y_pred)
        cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=0)
        cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=1)   
        plt.figure()
        plot_confusion_matrix(cfMatrix, labels.listLabels)
        plt.show()

    # Write report to txt
    with open(reportPath,'w') as fileObj:
        fileObj.write(report)
        fileObj.write(str(meanIoU))
        #fileObj.write(str(meanIoUfromcf))
        
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


class TestCallback(Callback):
    def __init__(self, epochs, testgenerator, batchsize, testSetSize):
        self.score      = np.zeros([epochs,2])
        self.generator  = testgenerator
        self.batchsize  = batchsize
        self.set        = testSetSize

    def on_epoch_end(self, epoch, logs={}):           
        self.score[epoch,:] = self.model.evaluate_generator(self.generator,steps=self.set//self.batchsize,use_multiprocessing=True)
        print('\nTesting loss: {}, acc: {}\n'.format(self.score[epoch,0], self.score[epoch,1]))

    # Write report to txt
    '''
    with open(reportPath,'w') as fileObj:
        fileObj.write(report)
        fileObj.write(str(meanIoU))
        fileObj.write(str(meanIoUfromcf))
    '''
if __name__ == '__main__':
    print('This module cannot run explicitly.')
    