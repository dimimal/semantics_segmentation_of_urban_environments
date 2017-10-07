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

idLabels = { id:label for label,id in labels.iteritems()}
