#!/usr/bin/python
# -*- coding: utf-8 -*-


#################
## Import modules
#################

# pyqt for everything graphical
from PyQt4 import QtGui
from PyQt4 import QtCore
# get command line parameters
import sys
# walk directories
import glob
# access to OS functionality
import os
# call processes
import subprocess
# copy things
import copy 
# numpy
import numpy as np
# re for regular expression usage
import re
# Load Opencv to make resizes 
import cv2 as cv
# matplotlib for colormaps
try:
    import matplotlib.colors
    import matplotlib.cm
    from PIL import PILLOW_VERSION
    from PIL import Image
except:
    pass

#################
## Helper classes
#################

# Import relative modules
from helpers.annotation import Annotation
from helpers.labels import name2label, assureSingleInstanceName, trainId2color, trainId2label


#################
## Main GUI class
#################

# The main class which is a QtGui -> Main Window
class CityscapesViewer(QtGui.QMainWindow):

    #############################
    ## Construction / Destruction
    #############################

    # Constructor
    def __init__(self):
        # Construct base class
        super(CityscapesViewer, self).__init__()

        # This is the configuration.

        # The filename of the image we currently working on
        self.currentFile       = ""
        # The filename of the labels we currently working on
        self.currentLabelFile  = ""
        # The path of the images of the currently loaded city
        #self.city              = ""
        # The name of the currently loaded city
        #self.cityName          = ""
        # The path of the labels. In this folder we expect a folder for each city
        # Within these city folders we expect the label with a filename matching
        # the images, except for the extension
        #self.labelPath         = ""
        # The transparency of the labels over the image
        self.transp            = 0.5
        # The zoom toggle
        self.zoom              = False
        # The zoom factor
        self.zoomFactor        = 1.5
        # The size of the zoom window. Currently there is no setter or getter for that
        self.zoomSize          = 400 #px

        # The width that we actually use to show the image
        self.w                 = 0
        # The height that we actually use to show the image
        self.h                 = 0
        # The horizontal offset where we start drawing within the widget
        self.xoff              = 0
        # The vertical offset where we start drawing withing the widget
        self.yoff              = 0
        # A gap that we  leave around the image as little border
        self.bordergap         = 20
        # The scale that was used, ie
        #self.w = self.scale * self.image.width()
        #self.h = self.scale * self.image.height()
        self.scale             = 1.0
        # Filenames of all images in current city
        self.images            = []
        # Image extension
        self.imageExt          = "_leftImg8bit.png"
        # Ground truth extension
        #self.gtExt             = "_gt*_polygons.json"
        self.gtExt             = "_gt*_*.png"
        # Prediction Extension Pattern
        self.predExt           = "_predictions*.txt"
        # Current image as QImage
        self.image             = QtGui.QImage()
        # Index of the current image within the city folder
        #self.idx               = 0
        # All annotated objects in current image, i.e. list of labelObject
        self.annotation        = QtGui.QImage()
        # All prediction files that came from model
        self.predictions       = []
        # The current object the mouse points to. It's index in self.labels
        self.mouseObj          = -1
        # The object that is highlighted and its label. An object instance
        self.highlightObj      = None
        self.highlightObjLabel = None
        # The position of the mouse
        self.mousePosOrig      = None
        # The position of the mouse scaled to label coordinates
        self.mousePosScaled    = None
        # If the mouse is outside of the image
        self.mouseOutsideImage = True
        # The position of the mouse upon enabling the zoom window
        self.mousePosOnZoom    = None
        # A list of toolbar actions that need an image
        self.actImage          = []
        # A list of toolbar actions that need an image that is not the first
        self.actImageNotFirst  = []
        # A list of toolbar actions that need an image that is not the last
        self.actImageNotLast   = []
        # Toggle status of the play icon
        self.playState         = False
        # Enable disparity visu in general
        self.enableDisparity   = True
        # Show disparities instead of labels
        self.showDisparity     = False
        # The filename of the disparity map we currently working on
        self.currentDispFile   = ""
        # The disparity image
        self.dispImg           = None
        # As overlay
        self.dispOverlay       = None
        # The disparity search path
        self.dispPath          = None
        # Disparity extension
        self.dispExt           = "_disparity.png"
        # Generate colormap
        try:
            norm = matplotlib.colors.Normalize(vmin=3,vmax=100)
            cmap = matplotlib.cm.plasma
            self.colormap = matplotlib.cm.ScalarMappable( norm=norm , cmap=cmap )
        except:
            self.enableDisparity = False
        # check if pillow was imported, otherwise no disparity visu possible
        if not 'PILLOW_VERSION' in globals():
            self.enableDisparity = False

        # Default label
        self.defaultLabel = 'static'
        if self.defaultLabel not in name2label:
            print('The {0} label is missing in the internal label definitions.'.format(self.defaultLabel))
            return
        # Last selected label
        self.lastLabel = self.defaultLabel

        # Setup the GUI
        self.initUI()

        # Retrieve the evaluation mode
        #self.selectMode()

        # Load Model First (If we add model integration)
        self.loadModel()

        # If we already know a city from the saved config -> load it
        #self.loadCity()
        self.imageChanged()

    # Destructor
    def __del__(self):
        return

    # Construct everything GUI related. Called by constructor
    def initUI(self):
        # Create a toolbar
        self.toolbar = self.addToolBar('Tools')

        # Add the tool buttons
        iconDir = os.path.join( os.path.dirname(sys.argv[0]) , 'icons' )

        # Loading a new city
        loadAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'open.png' )), '&Tools', self)
        loadAction.setShortcuts(['o'])
        #self.setTip( loadAction, 'Open city' )
        #loadAction.triggered.connect( self.getCityFromUser )
        self.setTip(loadAction, 'Open Image')
        loadAction.triggered.connect( self.getImageFromUser)
        self.toolbar.addAction(loadAction)

        '''
        # Open previous image
        backAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'back.png')), '&Tools', self)
        backAction.setShortcut('left')
        backAction.setStatusTip('Previous image')
        backAction.triggered.connect( self.prevImage )
        self.toolbar.addAction(backAction)
        self.actImageNotFirst.append(backAction)
        # Open next image
        nextAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'next.png')), '&Tools', self)
        nextAction.setShortcut('right')
        self.setTip( nextAction, 'Next image' )
        nextAction.triggered.connect( self.nextImage )
        self.toolbar.addAction(nextAction)
        self.actImageNotLast.append(nextAction)
        '''

        # Play
        '''
        playAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'play.png')), '&Tools', self)
        playAction.setShortcut(' ')
        playAction.setCheckable(True)
        playAction.setChecked(False)
        self.setTip( playAction, 'Play all images' )
        playAction.triggered.connect( self.playImages )
        self.toolbar.addAction(playAction)
        self.actImageNotLast.append(playAction)
        self.playAction = playAction
        '''
        # Select image
        selImageAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'shuffle.png' )), '&Tools', self)
        selImageAction.setShortcut('i')
        self.setTip( selImageAction, 'Select image' )
        selImageAction.triggered.connect( self.selectImage )
        self.toolbar.addAction(selImageAction)
        self.actImage.append(selImageAction)
        '''
        # Enable/disable disparity visu. Toggle button
        if self.enableDisparity:
            dispAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'disp.png' )), '&Tools', self)
            dispAction.setShortcuts(['d'])
            dispAction.setCheckable(True)
            dispAction.setChecked(self.showDisparity)
            self.setTip( dispAction, 'Enable/disable depth visualization' )
            dispAction.toggled.connect( self.dispToggle )
            self.toolbar.addAction(dispAction)
            self.actImage.append(dispAction)
        '''
        # Enable/disable zoom. Toggle button
        zoomAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'zoom.png' )), '&Tools', self)
        zoomAction.setShortcuts(['z'])
        zoomAction.setCheckable(True)
        zoomAction.setChecked(self.zoom)
        self.setTip( zoomAction, 'Enable/disable permanent zoom' )
        zoomAction.toggled.connect( self.zoomToggle )
        self.toolbar.addAction(zoomAction)
        self.actImage.append(zoomAction)

        # Decrease transparency
        minusAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'minus.png' )), '&Tools', self)
        minusAction.setShortcut('-')
        self.setTip( minusAction, 'Decrease transparency' )
        minusAction.triggered.connect( self.minus )
        self.toolbar.addAction(minusAction)

        # Increase transparency
        plusAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'plus.png' )), '&Tools', self)
        plusAction.setShortcut('+')
        self.setTip( plusAction, 'Increase transparency' )
        plusAction.triggered.connect( self.plus )
        self.toolbar.addAction(plusAction)

        # Display path to current image in message bar
        displayFilepathAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'filepath.png' )), '&Tools', self)
        displayFilepathAction.setShortcut('f')
        self.setTip( displayFilepathAction, 'Show path to current image' )
        displayFilepathAction.triggered.connect( self.displayFilepath )
        self.toolbar.addAction(displayFilepathAction)

        # Display help message
        helpAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'help19.png' )), '&Tools', self)
        helpAction.setShortcut('h')
        self.setTip( helpAction, 'Help' )
        helpAction.triggered.connect( self.displayHelpMessage )
        self.toolbar.addAction(helpAction)

        # Close the application
        exitAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'exit.png' )), '&Tools', self)
        exitAction.setShortcuts(['Esc'])
        self.setTip( exitAction, 'Exit' )
        exitAction.triggered.connect( self.close )
        self.toolbar.addAction(exitAction)

        # The default text for the status bar
        self.defaultStatusbar = 'Ready'
        # Create a statusbar. Init with default
        self.statusBar().showMessage( self.defaultStatusbar )

        # Enable mouse move events
        self.setMouseTracking(True)
        self.toolbar.setMouseTracking(True)
        # Open in full screen
        self.showFullScreen( )
        # Set a title
        self.applicationTitle = 'Cityscapes Viewer v1.0'
        self.setWindowTitle(self.applicationTitle)
        self.displayHelpMessage()
        self.selectMode()
        self.getImageFromUser()
        #self.getCityFromUser()
        # And show the application
        self.show()

    #############################
    ## Toolbar call-backs
    #############################

    # Switch to previous image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def prevImage(self):
        if not self.images:
            return
        if self.idx > 0:
            self.idx -= 1
            self.imageChanged()
        else:
            message = "Already at the first image"
            self.statusBar().showMessage(message)
        return

    # Switch to next image in file list
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    '''
    def nextImage(self):
        if not self.images:
            return
        if self.idx < len(self.images)-1:
            self.idx += 1
            self.imageChanged()
        elif self.playState:
            self.playState = False
            self.playAction.setChecked(False)
        else:
            message = "Already at the last image"
            self.statusBar().showMessage(message)
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)
        return

    # Play images, i.e. auto-switch to next image
    def playImages(self, status):
        self.playState = status
        if self.playState:
            QtCore.QTimer.singleShot(0, self.nextImage)

    '''
    # Switch to a selected image of the file list
    # Ask the user for an image
    # Load the image
    # Load its labels
    # Update the mouse selection
    # View
    def selectImage(self):
        
        if not self.images:
            return
        
        dlgTitle = "Select image to load"
        self.statusBar().showMessage(dlgTitle)
        items = QtCore.QStringList( [ os.path.basename(i) for i in self.images ] )
        (item, ok) = QtGui.QInputDialog.getItem(self, dlgTitle, "Image", items, self.idx, False)
        if (ok and item):
            idx = items.indexOf(item)
            if idx != self.idx:
                self.idx = idx
                self.imageChanged()
        else:
            # Restore the message
            self.statusBar().showMessage( self.defaultStatusbar )


    # Toggle zoom
    def zoomToggle(self, status):
        self.zoom = status
        if status :
            self.mousePosOnZoom = self.mousePosOrig
        self.update()

    # Toggle disparity visu
    def dispToggle(self, status):
        self.showDisparity = status
        self.imageChanged()


    # Increase label transparency
    def minus(self):
        self.transp = max(self.transp-0.1,0.0)
        self.update()


    def displayFilepath(self):
        self.statusBar().showMessage("Current image: {0}".format( self.currentFile ))
        self.update()

    def displayHelpMessage(self):

        message = self.applicationTitle + "\n\n"
        message += "INSTRUCTIONS\n"
        message += " - select a city from drop-down menu\n"
        message += " - browse images and labels using\n"
        message += "   the toolbar buttons or the controls below\n"
        message += "\n"
        message += "CONTROLS\n"
        message += " - select city [o]\n"
        message += " - highlight objects [move mouse]\n"
        message += " - next image [left arrow]\n"
        message += " - previous image [right arrow]\n"
        message += " - toggle autoplay [space]\n"
        message += " - increase/decrease label transparency\n"
        message += "   [ctrl+mousewheel] or [+ / -]\n"
        message += " - open zoom window [z]\n"
        message += "       zoom in/out [mousewheel]\n"
        message += "       enlarge/shrink zoom window [shift+mousewheel]\n"
        message += " - select a specific image [i]\n"
        message += " - show path to image below [f]\n"
        message += " - exit viewer [esc]\n"

        QtGui.QMessageBox.about(self, "HELP!", message)
        self.update()


    # Decrease label transparency
    def plus(self):
        self.transp = min(self.transp+0.1,1.0)
        self.update()

    # Close the application
    def closeEvent(self,event):
         event.accept()


    #############################
    ## Custom events
    #############################

    def imageChanged(self):
        # Load the first image
        self.loadImage()
        # Load its labels if available
        #self.loadLabels()
        # Load disparities if available
        #self.loadDisparities()
        # Update the object the mouse points to
        self.updateMouseObject()
        # Update the GUI
        self.update()

    #############################
    ## File I/O
    #############################
    '''
    # Load the currently selected city if possible
    def loadCity(self):
        # Search for all *.pngs to get the image list
        self.images = []
        if os.path.isdir(self.city):
            self.images = glob.glob( os.path.join( self.city , '*' + self.imageExt ) )
            self.images.sort()
            if self.currentFile in self.images:
                self.idx = self.images.index(self.currentFile)
            else:
                self.idx = 0
    '''
    # Instead of loadCity method loadImages introduced to load images directly
    def loadImages(self):
        #self.images      = []
        #if os.path.isdir(self.imagePath):
        #self.images      = image
        #self.images.sort()
        #self.annotations = predictions
        #self.idx  = self.images.index()
        ''' 
        if self.currentFile in self.images:
            self.idx = self.images.index(self.currentFile)
        else:
            self.idx = 0
        self.annotations = glob.glob('*' + self.gtExt)
        self.image
        '''
        #print('line 499', type(self.images))
        #test = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/modelViewer/aachen_000000_000019_leftImg8bit.png'
        '''
        print(str(self.images))
        if os.path.exists(str(self.images)):
            print('welcome')
        '''
        self.image  = QtGui.QImage(self.images)
        self.idx    = 0

        self.loadLabels()
        self.loadImage()
        #self.images      = image
        #self.annotations = predictions

    # Load the currently selected image
    # Does only load if not previously loaded
    # Does not refresh the GUI
    def loadImage(self):
        success = False
        message = self.defaultStatusbar
        if self.images:
            #filename = self.images[self.idx]
            #filename = os.path.normpath( filename )
            if not self.image.isNull():# and filename == self.currentFile:
                success = True
            else:
                self.image = QtGui.QImage(self.images)
                if self.image.isNull():
                    message = "Failed to read image: {0}".format( str(self.images) )
                else:
                    message = "Read image: {0}".format( str(self.images) )
                    self.currentFile = str(self.images)
                    success = True

        # Update toolbar actions that need an image
        for act in self.actImage:
            act.setEnabled(success)
        for act in self.actImageNotFirst:
            act.setEnabled(success and self.idx > 0)
        for act in self.actImageNotLast:
            act.setEnabled(success and self.idx < len(self.images)-1)

        self.statusBar().showMessage(message)

    # Show the dialog box to select the proper mode prediction mode
    # In progress:: TODO: Add properly the functionalities
    def selectMode(self):
        window  = QtGui.QWidget()
        layout  = QtGui.QHBoxLayout()
        self.b1 = QtGui.QCheckBox("Evalutaion")
        self.b2 = QtGui.QCheckBox('Prediction')
        self.b1.setChecked(True)
        ok      = QtGui.QPushButton(window)
        ok.setText('OK')
        self.bg = QtGui.QButtonGroup()
        self.bg.addButton(self.b1, 1) 
        self.bg.addButton(self.b2, 1) 
        #self.b1.stateChanged.connect(lambda:self.btnstate(self.b1))
        layout.addWidget(self.b1)
        layout.addWidget(self.b2)
        layout.addStretch()
        self.bg.buttonClicked[QtGui.QAbstractButton].connect(self.btngroup)    

        window.setLayout(layout)
        window.setWindowTitle('Select tool mode')
        #self.setWindowTitle("Select evaluation or prediction")
        
        window.show()
        self.update()
            
    def btngroup(self, button):
        if button.text() == 'Evaluation':
            self.mode = 'Eval'
        elif button.text() == 'Prediction':
            self.mode = 'Prediction'

    # Load the model from file to get predictions
    def loadModel(self):
        pass

    # Check the dimensions of the labels to match with the image
    def checkDims(self):
        if self.annotations.ndim == 1:
            self.annotations = np.reshape(self.annotations, (512,512))
        
        # Find the scale factors of the image
        fx               = 1024 / self.annotations.shape[0]
        fy               = 2048 / self.annotations.shape[1]
        shape = self.annotations.shape

        self.annotations = cv.resize(self.annotations, (fy*shape[0], fx*shape[1]), interpolation=cv.INTER_NEAREST)
    
    # Transform trainId labels to colors
    def labels2Color(self):
        temp = np.empty((self.annotations.shape[0], self.annotations.shape[1], 3), dtype='uint8')
        for index, value in np.ndenumerate(self.annotations):
            temp[index[0], index[1]] = trainId2color[value]
        self.annotation = temp
            
    # The new loadLabels Function
    def loadLabels(self):
        #print(self.annotations)

        self.annotations = np.loadtxt(str(self.annotations))
        self.annotations = np.asarray(self.annotations, dtype='uint8')
        self.checkDims()
        self.labels2Color()

        # Labels to QImage
        self.annotation  = self.toQImage(self.annotation)

        # Remeber the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage( restoreMessage )
    
    # Load the labels from file
    # Only loads if they exist
    # Otherwise the filename is stored and that's it
        '''
    def loadLabels(self):
        
        filename = self.getLabelFilename()
        if not filename:
            self.clearAnnotation()
            return

        # If we have everything and the filename did not change, then we are good
        if self.annotation and filename == self.currentLabelFile:
            return

        # Clear the current labels first
        self.clearAnnotation()

        try:
            self.annotation = Annotation()
            self.annotation.fromJsonFile(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing labels in {0}. Message: {1}".format( filename, e.strerror )
            self.statusBar().showMessage(message)

        # Remember the filename loaded
        self.currentLabelFile = filename

        # Remeber the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage( restoreMessage )
        '''
    
    '''
    # Load the disparity map from file
    # Only loads if they exist
    def loadDisparities(self):
        if not self.enableDisparity:
            return
        if not self.showDisparity:
            return
        
        filename = self.getDisparityFilename()
        if not filename:
            self.dispImg = None
            return
        
        # If we have everything and the filename did not change, then we are good
        if self.dispImg and filename == self.currentDispFile:
            return

        # Clear the current labels first
        self.dispImg = None

        try:
            self.dispImg = Image.open(filename)
        except IOError as e:
            # This is the error if the file does not exist
            message = "Error parsing disparities in {0}. Message: {1}".format( filename, e.strerror )
            self.statusBar().showMessage(message)
            self.dispImg = None

        if self.dispImg:
            dispNp = np.array( self.dispImg )
            dispNp /= 128
            dispNp.round()
            dispNp = np.array( dispNp , dtype=np.uint8 )

            dispQt = QtGui.QImage( dispNp.data , dispNp.shape[1] , dispNp.shape[0] , QtGui.QImage.Format_Indexed8 )

            colortable = []
            for i in range(256):
                color = self.colormap.to_rgba(i)
                colorRgb = ( int(color[0]*255) , int(color[1]*255) , int(color[2]*255) )
                colortable.append( QtGui.qRgb( *colorRgb ) )

            dispQt.setColorTable( colortable )
            dispQt = dispQt.convertToFormat( QtGui.QImage.Format_ARGB32_Premultiplied )
            self.dispOverlay = dispQt

        # Remember the filename loaded
        self.currentDispFile = filename

        # Remember the status bar message to restore it later
        restoreMessage = self.statusBar().currentMessage()

        # Restore the message
        self.statusBar().showMessage( restoreMessage )
    '''
    #############################
    ## Drawing
    #############################

    # This method is called when redrawing everything
    # Can be manually triggered by self.update()
    # Note that there must not be any other self.update within this method
    # or any methods that are called within
    def paintEvent(self, event):
        # Create a QPainter that can perform draw actions within a widget or image
        qp = QtGui.QPainter()
        # Begin drawing in the application widget
        qp.begin(self)
        # Update scale
        self.updateScale(qp)
        # Determine the object ID to highlight
        #self.getHighlightedObject(qp)
        # Draw the image first
        self.drawImage(qp)
        '''
        if self.enableDisparity and self.showDisparity:
            # Draw the disparities on top
            overlay = self.drawDisp(qp)
        else:
        '''
        # Draw the labels on top
        overlay = self.drawLabels(qp)
        # Draw the label name next to the mouse
        self.drawLabelAtMouse(qp)
        
        # Draw the zoom
        self.drawZoom(qp, overlay)

        # Thats all drawing
        qp.end()

        # Forward the paint event
        QtGui.QMainWindow.paintEvent(self,event)

    # Update the scaling
    def updateScale(self, qp):
        if not self.image.width() or not self.image.height():
            return
        # Horizontal offset
        self.xoff  = self.bordergap
        # Vertical offset
        self.yoff  = self.toolbar.height()+self.bordergap
        # We want to make sure to keep the image aspect ratio and to make it fit within the widget
        # Without keeping the aspect ratio, each side of the image is scaled (multiplied) with
        sx = float(qp.device().width()  - 2*self.xoff) / self.image.width()
        sy = float(qp.device().height() - 2*self.yoff) / self.image.height()
        # To keep the aspect ratio while making sure it fits, we use the minimum of both scales
        # Remember the scale for later
        self.scale = min( sx , sy )
        # These are then the actual dimensions used
        self.w     = self.scale * self.image.width()
        self.h     = self.scale * self.image.height()

    # Determine the highlighted object for drawing
    def getHighlightedObject(self, qp):
        # This variable we want to fill
        self.highlightObj = None

        # Without labels we cannot do so
        if not self.annotation:
            return

        # If available its the selected object
        highlightObjId = -1
        # If not available but the polygon is empty or closed, its the mouse object
        if highlightObjId < 0 and not self.mouseOutsideImage:
            highlightObjId = self.mouseObj
        # Get the actual object that is highlighted
        if highlightObjId >= 0:
            self.highlightObj = self.annotation.objects[highlightObjId]
            self.highlightObjLabel = self.annotation.objects[highlightObjId].label

    # Draw the image in the given QPainter qp
    def drawImage(self, qp):
        # Return if no image available
        if self.image.isNull():
            return

        # Save the painters current setting to a stack
        qp.save()
        # Draw the image
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), self.image)
        # Restore the saved setting from the stack
        qp.restore()
    '''
    def getPolygon(self, obj):
        poly = QtGui.QPolygonF()
        for pt in obj.polygon:
            point = QtCore.QPointF(pt.x,pt.y)
            poly.append( point )
        return poly
    '''
    # Draw the labels in the given QPainter qp
    # optionally provide a list of labels to ignore
    def drawLabels(self, qp, ignore = []):
        if self.image.isNull() or self.w == 0 or self.h == 0:
            return
        if self.annotation.isNull():
            return
        '''
        if not self.annotation:
            return
        '''

        # The overlay is created in the viewing coordinates
        # This way, the drawing is more dense and the polygon edges are nicer
        # We create an image that is the overlay
        # Within this image we draw using another QPainter
        # Finally we use the real QPainter to overlay the overlay-image on what is drawn so far

        # The image that is used to draw the overlays
        #overlay = QtGui.QImage( self.w, self.h, QtGui.QImage.Format_ARGB32_Premultiplied )
        # Fill the image with the default color
        #defaultLabel = name2label[self.defaultLabel]
        #col = QtGui.QColor( *defaultLabel.color )
        #overlay.fill( col )

        # Create a new QPainter that draws in the overlay image
        qp2 = QtGui.QPainter()
        #qp2.begin(overlay)
        qp2.begin(self.annotation)
        
        # Save the painters current setting to a stack
        qp.save()
        # Draw the image
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), self.image)
        #qp2.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), self.annotation)
        
        # Restore the saved setting from the stack
        qp.restore()

        # The color of the outlines
        #qp2.setPen(QtGui.QColor('white'))
        '''
        # Draw all objects
        for obj in self.annotation.objects:

            # The label of the object
            name      = assureSingleInstanceName( obj.label )
            # If we do not know a color for this label, warn the user
            if name not in name2label:
                print("The annotations contain unkown labels. This should not happen. Please inform the datasets authors. Thank you!")
                print("Details: label '{}', file '{}'".format(name,self.currentLabelFile))
                continue

            #poly = self.getPolygon(obj)

            # Scale the polygon properly
            #polyToDraw = poly * QtGui.QTransform.fromScale(self.scale,self.scale)

            # Default drawing
            # Color from color table, solid brush
            col   = QtGui.QColor( *name2label[name].color     )
            brush = QtGui.QBrush( col, QtCore.Qt.SolidPattern )
            qp2.setBrush(brush)
            # Overwrite drawing if this is the highlighted object
            if self.highlightObj and obj == self.highlightObj:
                # First clear everything below of the polygon
                qp2.setCompositionMode( QtGui.QPainter.CompositionMode_Clear )
                #qp2.drawPolygon( polyToDraw )
                qp2.setCompositionMode( QtGui.QPainter.CompositionMode_SourceOver )
                # Set the drawing to a special pattern
                brush = QtGui.QBrush(col,QtCore.Qt.DiagCrossPattern)
                qp2.setBrush(brush)

            qp2.drawPolygon( polyToDraw )
        # Draw outline of selected object dotted
        if self.highlightObj:
            brush = QtGui.QBrush(QtCore.Qt.NoBrush)
            qp2.setBrush(brush)
            qp2.setPen(QtCore.Qt.DashLine)
            polyToDraw = self.getPolygon(self.highlightObj) * QtGui.QTransform.fromScale(self.scale,self.scale)
            qp2.drawPolygon( polyToDraw )

        '''
        # End the drawing of the overlay
        qp2.end()
        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        #qp.drawImage(self.xoff, self.yoff, overlay)
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ), self.annotation)
        # Restore settings
        qp.restore()

        return self.annotation
    
    # Numpy array to QImage 
    def toQImage(self, im, copy=False):
        if im is None:
            return QtGui.QImage()
        
        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                gray_color_table = [qRgb(i, i, i) for i in range(256)]
                qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QtGui.QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_ARGB32)
                    return qim.copy() if copy else qim
        # Add failure message here!!

    # Draw the label name next to the mouse
    def drawLabelAtMouse(self, qp):
        # Nothing to do without a highlighted object
        '''
        if not self.highlightObj:
            return
        if not self.annotations.all():
            return
        '''
        # Nothing to without a mouse position
        if not self.mousePosOrig:
            return

        # Save QPainter settings to stack
        qp.save()

        # That is the mouse positiong
        mouse = self.mousePosOrig
        #print(mouse.x(), mouse.y())
        #print(self.annotations.shape)
        # Will show zoom
        showZoom = self.zoom and not self.image.isNull() and self.w and self.h

        #mouseText = self.highlightObj.label
        # The label of current cursor position
        trainIdIndex  = self.annotations[int(mouse.y()), int(mouse.x())]
        # The text that is written next to the mouse
        mouseText     = trainId2label[trainIdIndex][0]

        # Where to write the text
        # Depends on the zoom (additional offset to mouse to make space for zoom?)
        # The location in the image (if we are at the top we want to write below of the mouse)
        off = 36
        if showZoom:
            off += self.zoomSize/2
        if mouse.y()-off > self.toolbar.height():
            top = mouse.y()-off
            btm = mouse.y()
            vAlign = QtCore.Qt.AlignTop
        else:
            # The height of the cursor
            if not showZoom:
                off += 20
            top = mouse.y()
            btm = mouse.y()+off
            vAlign = QtCore.Qt.AlignBottom

        # Here we can draw
        rect = QtCore.QRect()
        rect.setTopLeft(QtCore.QPoint(mouse.x()-200,top))
        rect.setBottomRight(QtCore.QPoint(mouse.x()+200,btm))

        # The color
        qp.setPen(QtGui.QColor('white'))
        # The font to use
        font = QtGui.QFont("Helvetica",20,QtGui.QFont.Bold)
        qp.setFont(font)
        # Non-transparent
        qp.setOpacity(1)
        # Draw the text, horizontally centered
        qp.drawText(rect,QtCore.Qt.AlignHCenter|vAlign,mouseText)
        # Restore settings
        qp.restore()

    # Draw the zoom
    def drawZoom(self,qp,overlay):
        # Zoom disabled?
        if not self.zoom:
            return
        # No image
        if self.image.isNull() or not self.w or not self.h:
            return
        # No mouse
        if not self.mousePosOrig:
            return

        # Abbrevation for the zoom window size
        zoomSize = self.zoomSize
        # Abbrevation for the mouse position
        mouse = self.mousePosOrig

        # The pixel that is the zoom center
        pix = self.mousePosScaled
        # The size of the part of the image that is drawn in the zoom window
        selSize = zoomSize / ( self.zoomFactor * self.zoomFactor )
        # The selection window for the image
        sel  = QtCore.QRectF(pix.x()  -selSize/2 ,pix.y()  -selSize/2 ,selSize,selSize  )
        # The selection window for the widget
        view = QtCore.QRectF(mouse.x()-zoomSize/2,mouse.y()-zoomSize/2,zoomSize,zoomSize)
        if overlay :
            overlay_scaled = overlay.scaled(self.image.width(), self.image.height())
        else :
            overlay_scaled = QtGui.QImage( self.image.width(), self.image.height(), QtGui.QImage.Format_ARGB32_Premultiplied )

        # Show the zoom image
        qp.save()
        qp.drawImage(view,self.image,sel) # Original Image
        qp.setOpacity(self.transp)
        qp.drawImage(view,overlay_scaled,sel) # Label Image
        qp.restore()

    '''
    # Draw disparities
    def drawDisp( self , qp ):
        if not self.dispOverlay:
            return 

        # Save QPainter settings to stack
        qp.save()
        # Define transparency
        qp.setOpacity(self.transp)
        # Draw the overlay image
        qp.drawImage(QtCore.QRect( self.xoff, self.yoff, self.w, self.h ),self.dispOverlay)
        # Restore settings
        qp.restore()

        return self.dispOverlay
    '''

    #############################
    ## Mouse/keyboard events
    #############################

    # Mouse moved
    # Need to save the mouse position
    # Need to drag a polygon point
    # Need to update the mouse selected object
    def mouseMoveEvent(self,event):
        if self.image.isNull() or self.w == 0 or self.h == 0:
            return

        mousePosOrig = QtCore.QPointF( event.x() , event.y() )
        mousePosScaled = QtCore.QPointF( float(mousePosOrig.x() - self.xoff) / self.scale , float(mousePosOrig.y() - self.yoff) / self.scale )
        mouseOutsideImage = not self.image.rect().contains( mousePosScaled.toPoint() )

        mousePosScaled.setX( max( mousePosScaled.x() , 0. ) )
        mousePosScaled.setY( max( mousePosScaled.y() , 0. ) )
        mousePosScaled.setX( min( mousePosScaled.x() , self.image.rect().right() ) )
        mousePosScaled.setY( min( mousePosScaled.y() , self.image.rect().bottom() ) )

        if not self.image.rect().contains( mousePosScaled.toPoint() ):
            print(self.image.rect())
            print(mousePosScaled.toPoint())
            self.mousePosScaled = None
            self.mousePosOrig = None
            self.updateMouseObject()
            self.update()
            return

        self.mousePosScaled    = mousePosScaled
        self.mousePosOrig      = mousePosOrig
        self.mouseOutsideImage = mouseOutsideImage

        # Redraw
        self.updateMouseObject()
        self.update()

    # Mouse left the widget
    def leaveEvent(self, event):
        self.mousePosOrig = None
        self.mousePosScaled = None
        self.mouseOutsideImage = True


    # Mouse wheel scrolled
    def wheelEvent(self, event):
        ctrlPressed = event.modifiers() & QtCore.Qt.ControlModifier

        deltaDegree = event.delta() / 8 # Rotation in degree
        deltaSteps  = deltaDegree / 15 # Usually one step on the mouse is 15 degrees

        if ctrlPressed:
            self.transp = max(min(self.transp+(deltaSteps*0.1),1.0),0.0)
            self.update()
        else:
            if self.zoom:
                # If shift is pressed, change zoom window size
                if event.modifiers() and QtCore.Qt.Key_Shift:
                    self.zoomSize += deltaSteps * 10
                    self.zoomSize = max( self.zoomSize, 10   )
                    self.zoomSize = min( self.zoomSize, 1000 )
                # Change zoom factor
                else:
                    self.zoomFactor += deltaSteps * 0.05
                    self.zoomFactor = max( self.zoomFactor, 0.1 )
                    self.zoomFactor = min( self.zoomFactor, 10 )
                self.update()


    #############################
    ## Little helper methods
    #############################

    # Helper method that sets tooltip and statustip
    # Provide an QAction and the tip text
    # This text is appended with a hotkeys and then assigned
    def setTip( self, action, tip ):
        tip += " (Hotkeys: '" + "', '".join([str(s.toString()) for s in action.shortcuts()]) + "')"
        action.setStatusTip(tip)
        action.setToolTip(tip)

    # Update the object that is selected by the current mouse curser
    # Edw mporw na valc na fainontai ta annotated labels
    def updateMouseObject(self):
        self.mouseObj   = -1
        if self.mousePosScaled is None:
            return
        '''
        for idx in reversed(range(len(self.annotation.objects))):
            obj = self.annotation.objects[idx]
            if self.getPolygon(obj).containsPoint(self.mousePosScaled, QtCore.Qt.OddEvenFill):
                self.mouseObj = idx
                break
        '''

    # Clear the current labels
    def clearAnnotation(self):
        self.annotation = None
        self.currentLabelFile = ""
    
    def getFile(self):
        annotations      = [ "gtFine" ]
        rawImagePatttern = ["leftImg8bit"]

        dir_path    = os.path.dirname(os.path.realpath(__file__))
        message     = 'Select image to get predictions' 
        self.statusBar().showMessage(message)
        fileDialog  = QtGui.QFileDialog.getOpenFileName(self, 'Select Image for prediction', 
                                                        dir_path, 'Image files (*.jpg *.png *.gif)')
        # Put some error message here
        # fix dat shit!
        if rawImagePatttern[0] not in fileDialog:
            self.getFile()
        else:
            filePred = QtGui.QFileDialog.getOpenFileName(self, 'Load the prediction File', 
                                                        dir_path, 'Text Files or Image Files (*.txt)')
        
        return fileDialog, filePred
    
    # Replace getCityFromUser and load only one image
    def getImageFromUser(self):
        restoreMessage   = self.statusBar().currentMessage()

        annotations      = [ "gtFine" ]
        rawImagePatttern = [ "leftImg8bit" ]

        self.images, self.annotations  = self.getFile()
        
        self.loadImages()
        self.imageChanged()
        '''
        dlgTitle = "Select new Image for Predictions"
        message  = dlgTitle
        question = dlgTitle
        message  = "Select image for viewing"
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.statusBar().showMessage(message)
        #for file in os.listdir(dir_path)
        #print(dir_path)
        items = [file for file in os.listdir(dir_path) if annotations[0] in file or rawImagePatttern[0] in file]
        if items:

            (item, ok) = QtGui.QInputDialog.getItem(self, dlgTitle, question, items, 0, False)       
            # Restore message
            self.statusBar().showMessage(restoreMessage)

            if ok and item:
                
                # Check if test folder in order to load image only!! Test has no labels
                (split,gt,city) = [ str(i) for i in item.split(', ') ]
                if split == 'test' and not self.showDisparity:
                    self.transp = 0.1
                else:
                    self.transp = 0.5
                
                #print(str(items))
                self.transp = 0.5
                #print(os.path.normpath(dir_path))
                self.imagePath  = dir_path#+'/'+item 
                #pattern = re.findall('\w+_\d+_\d+_', str(item))[0] 
                self.labelPath  = dir_path# + '/' + pattern + annotations[0] + '_labelTrainIds.png'
                # If files not in place show proper message
                if not os.path.exists(self.imagePath) or not os.path.exists(self.labelPath):
                    message  = QtGui.QMessageBox()
                    message.setIcon(QtGui.QMessageBox.Warning)
                    message.setText('Path or file does not exist')
                    message.setInformativeText("Check if your files are in proper place or format")
                    message.setWindowTitle('Warning Message')
                    message.setStandardButtons(QtGui.QMessageBox.Ok)
                    sys.exit(message.exec_())
                # transform loadCity method to load image callback (loadImages())
                self.loadImages()
                self.imageChanged()
        else:
 
            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the folder is in the Cityscapes root folder\n"

            # Display wrong message
            reply = QtGui.QMessageBox.information(self, "ERROR!", warning, QtGui.QMessageBox.Ok)
            if reply == QtGui.QMessageBox.Ok:
                sys.exit()

        return                
    '''
    '''
    def getCityFromUser(self):
        # Reset the status bar to this message when leaving
        restoreMessage = self.statusBar().currentMessage()

        if 'CITYSCAPES_DATASET' in os.environ:
            csPath = os.environ['CITYSCAPES_DATASET']
        else:
            csPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')

        availableCities = []
        annotations = [ "gtFine" , "gtCoarse" ]
        splits      = [ "train_extra" , "train"  , "val" , "test" ]
        for gt in annotations:
            for split in splits:
                cities = glob.glob(os.path.join(csPath, gt, split, '*'))
                cities.sort()
                availableCities.extend( [ (split,gt,os.path.basename(c)) for c in cities if os.listdir(c) ] )

        # List of possible labels
        items = [split + ", " + gt + ", " + city for (split,gt,city) in availableCities]

        # Specify title
        dlgTitle = "Select new city"
        message  = dlgTitle
        question = dlgTitle
        message  = "Select city for viewing"
        question = "Which city would you like to view?"
        self.statusBar().showMessage(message)

        if items:

		    # Create and wait for dialog
		    (item, ok) = QtGui.QInputDialog.getItem(self, dlgTitle, question, items, 0, False)

		    # Restore message
		    self.statusBar().showMessage( restoreMessage )

		    if ok and item:
		        (split,gt,city) = [ str(i) for i in item.split(', ') ]
		        if split == 'test' and not self.showDisparity:
		            self.transp = 0.1
		        else:
		            self.transp = 0.5
		        self.city      = os.path.normpath( os.path.join( csPath, "leftImg8bit" , split , city ) )
		        self.labelPath = os.path.normpath( os.path.join( csPath, gt            , split , city ) )
		        self.dispPath  = os.path.normpath( os.path.join( csPath, "disparity"   , split , city ) )
		        self.loadCity()
		        self.imageChanged()

        else:
 
            warning = ""
            warning += "The data was not found. Please:\n\n"
            warning += " - make sure the scripts folder is in the Cityscapes root folder\n"
            warning += "or\n"
            warning += " - set CITYSCAPES_DATASET to the Cityscapes root folder\n"
            warning += "       e.g. 'export CITYSCAPES_DATASET=<root_path>'\n"

            reply = QtGui.QMessageBox.information(self, "ERROR!", warning, QtGui.QMessageBox.Ok)
            if reply == QtGui.QMessageBox.Ok:
                sys.exit()

        return
    '''
    '''
    # Determine if the given candidate for a label path makes sense
    def isLabelPathValid(self,labelPath):
        return os.path.isdir(labelPath)

    # Get the filename where to load labels
    # Returns empty string if not possible
    def getLabelFilename( self ):
        # And we need to have a directory where labels should be searched
        if not self.labelPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not self.isLabelPathValid(self.labelPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename( self.currentFile )
        filename = filename.replace( self.imageExt , self.gtExt )
        filename = os.path.join( self.labelPath , filename )
        search   = glob.glob( filename )
        if not search:
            return ""
        filename = os.path.normpath(search[0])
        return filename

    # Get the filename where to load disparities
    # Returns empty string if not possible
    def getDisparityFilename( self ):
        # And we need to have a directory where disparities should be searched
        if not self.dispPath:
            return ""
        # Without the name of the current images, there is also nothing we can do
        if not self.currentFile:
            return ""
        # Check if the label directory is valid.
        if not os.path.isdir(self.dispPath):
            return ""

        # Generate the filename of the label file
        filename = os.path.basename( self.currentFile )
        #filename = filename.replace( self.imageExt , self.dispExt )
        #filename = os.path.join( self.dispPath , filename )
        filename = os.path.normpath(filename)
        return filename
    '''

    # Disable the popup menu on right click
    def createPopupMenu(self):
        pass


def main():

    app = QtGui.QApplication(sys.argv)
    tool = CityscapesViewer()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
