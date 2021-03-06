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


        # Select image
        selImageAction = QtGui.QAction(QtGui.QIcon( os.path.join( iconDir , 'shuffle.png' )), '&Tools', self)
        selImageAction.setShortcut('i')
        self.setTip( selImageAction, 'Select image' )
        selImageAction.triggered.connect( self.selectImage )
        self.toolbar.addAction(selImageAction)
        self.actImage.append(selImageAction)
        
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
        self.showFullScreen()
        # Set a title
        self.applicationTitle = 'Cityscapes Prediction Viewer '
        self.setWindowTitle(self.applicationTitle)
        self.displayHelpMessage()
        self.selectMode()
        self.getImageFromUser()

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
        message += " - select an image along with the predictions\n"
        message += " - browse images and labels using\n"
        message += "   the toolbar buttons or the controls below\n"
        message += "\n"
        message += "CONTROLS\n"
        message += " - highlight objects [move mouse]\n"
        message += " - increase/decrease label transparency\n"
        message += "   [ctrl+mousewheel] or [+ / -]\n"
        message += " - open zoom window [z]\n"
        message += "       zoom in/out [mousewheel]\n"
        message += "       enlarge/shrink zoom window [shift+mousewheel]\n"
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
        # Update the object the mouse points to
        self.updateMouseObject()
        # Update the GUI
        self.update()

    def loadImages(self):
        """Load the images
        """
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
            if not self.image.isNull():
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
        """TODO: Load keras model
        """
        pass

    # Check the dimensions of the labels to match with the image
    def checkDims(self):
        if self.annotations.ndim == 1:
            if self.annotations == 1024*2048:
                self.annotations = np.reshape(self.annotations, (1024,2048))
            elif self.annotations == 512*512:
                self.annotations = np.reshape(self.annotations, (512,512))
                self.annotations = cv.resize(self.annotations, (2048, 1024), interpolation=cv.INTER_NEAREST)
        
        elif self.annotations.ndim == 2:
            if self.annotations.shape != (1024,2048):            
                self.annotations = cv.resize(self.annotations, (2048, 1024), interpolation=cv.INTER_NEAREST) 
    
    # Transform trainId labels to colors
    def labels2Color(self):
        temp = np.empty((self.annotations.shape[0], self.annotations.shape[1], 3), dtype='uint8')
        for index, value in np.ndenumerate(self.annotations):
            temp[index[0], index[1]] = trainId2color[value]
        self.annotation = temp
            
    # The new loadLabels Function
    def loadLabels(self):
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
        
        # Show Error message
        message = 'Inconsistent data type {}'.format(im.dtype)
        self.messageBox(message)
        # Update status bar
        self.statusBar().showMessage( self.defaultStatusbar )

        
    # Draw the label name next to the mouse
    def drawLabelAtMouse(self, qp):
        
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
        
    # Clear the current labels
    # 
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
        # Check if sth relevant selected
        if rawImagePatttern[0] not in fileDialog:
            message = 'Select an image'
            self.messageBox(message)
            return None, None
        else:
            filePred = QtGui.QFileDialog.getOpenFileName(self, 'Load the prediction File', 
                                                        dir_path, 'Text Files or Image Files (*.txt)')
        
            return fileDialog, filePred
    
    # Display message inside a box
    def messageBox(self, message):
        msg = QtGui.QMessageBox()
        msg.setIcon(QtGui.QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QtGui.QMessageBox.Ok)
        msg.exec_()

    # Replace getCityFromUser and load only one image
    def getImageFromUser(self):
        restoreMessage   = self.statusBar().currentMessage()

        annotations      = [ "gtFine" ]
        rawImagePatttern = [ "leftImg8bit" ]

        self.images, self.annotations  = self.getFile()

        if self.images and self.annotations:
            self.loadImages()
            self.imageChanged()      

    # Disable the popup menu on right click
    def createPopupMenu(self):
        pass


def main():

    app = QtGui.QApplication(sys.argv)
    tool = CityscapesViewer()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
