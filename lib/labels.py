#!/usr/bin/python

# This script has the labels for our dataset
# labels 		: 	 {class : id}     access by its class name 
# idlabels 		:    {id : class}   access by its label id
# listLabels	:    A list with the class labels with the same order

listLabels = [
			'road'			   	   , 
			'sidewalk'			   ,
			'building'             ,
			'wall'                 ,
			'fence'                ,
			'pole'                 ,
			'traffic light'        ,
			'traffic sign'         ,
			'vegetation'           ,
			'terrain'              ,
			'sky'                  ,
			'person'               ,
			'rider'                ,
			'car'                  ,
			'truck'                ,
			'bus'  				   ,
			'train'				   ,
			'motorcycle'		   ,
			'bicycle' 			   
			]

labels = {
			'road'				   :	0    , 
			'sidewalk'			   :	1    ,
			'building'             :    2    ,
			'wall'                 :    3    ,
			'fence'                :    4    ,
			'pole'                 :    5    ,
			'traffic light'        :    6    ,
			'traffic sign'         :    7    ,
			'vegetation'           :    8    ,
			'terrain'              :    9    ,
			'sky'                  :   10    ,
			'person'               :   11    ,
			'rider'                :   12    ,
			'car'                  :   13    ,
			'truck'                :   14    ,
			'bus'                  :   15    ,
			'train'                :   16    ,
			'motorcycle'           :   17    ,
			'bicycle'              :   18	 
		}
colors = {
    0 :(128, 64,128),
    1 : (244, 35,232),
    2 : ( 70, 70, 70),
    3 : (102,102,156),
    4 : (190,153,153),
    5 : (153,153,153),
    6 : (250,170, 30),
    7 : (220,220,  0),
    8 : (107,142, 35),
    9 : (152,251,152),
    10 :( 70,130,180),
    11 : (220, 20, 60),
    12 : (255,  0,  0),
    13 : (  0,  0,142),
    14 : (  0,  0, 70),
    15 : (  0, 60,100),
    16 : (  0, 80,100),
    17 : (  0,  0,230),
    18 : (119, 11, 32),
    19 : (255,255,255)}

classids = {
    0 	: 7,
    1 	: 8,
    2 	: 11,
    3 	: 12,
    4 	: 13,
    5 	: 17,
    6 	: 19,
    7 	: 20,
    8 	: 21,
    9 	: 22,
    10 	: 23,
    11 	: 24,
    12 	: 25,
    13 	: 26,
    14 	: 27,
    15 	: 28,
    16 	: 31,
    17 	: 32,
    18 	: 33,
    19 	: 0 }

idLabelsdict  	= { id:label 	for label,id in labels.iteritems()}
idLabelsList  	= [ id 		for label,id in labels.iteritems()]
trainId2Color 	= [ color  	for id,color in colors.iteritems()] 
trainId2ClassId = [ id		for trainId, id in classids.iteritems()]