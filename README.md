# Semantics Segmentation of Urban Environments

This is my undergraduate dissertation project. The goal of this thesis is to examine and compare the results from two variations of CNN Encode-Decode arhitectures using [Self-Normalization](https://arxiv.org/abs/1706.02515) technique along with CRF-RNN post processing unit. Due to visualize the results of the model properly a Visualizer based on [CityscapesScripts](https://github.com/mcordts/cityscapesScripts) has been implemented to visualize the results.

## Cityscapes Dataset
[Cityscapes](https://www.cityscapes-dataset.com/)


### Dependencies

* python 		2.7
* keras 		2.1 
* tensorflow 	1.4
* scikit-learn	0.19
* openCV		2.4
* numpy			1.13
* scipy			0.13
* pyQt4 for the Visualizer

Run `pip install -r requirements.txt` to intall the dependencies

# Arguments

`train.py [-h] [-n NETWORK] [-trp TRAINPATH] [-vdp VALIDATIONPATH]
                [-tsp TESTPATH] [-bs BATCHSIZE] [-crf] [-w [WEIGHTS]]
                [-m [MODEL]] [-e EPOCHS]`


## Results
![Input Image](https://github.com/dimimal/semantics_segmentation_of_urban_environments/blob/master/test_images/all_in_one.png)

### Installation
Run `make` inside `lib/crfasrnn_keras/src/cpp` to build highdimfilter module.
Create the npy data files for the data generator using `denseExtraction.py`.

Check the examples below to train your model.


### Examples
##### Training
`python train.py -n bdcnn -trp trainpath -vdp validationpath -tsp testpath -bs 4 -crf -e 20` 

##### Resume  Training
`python train.py -trp trainpath -vdp validationpath -tsp testpath -bs 4 -w weightspath -m modelpath -e 20`

## Acknowledgments

* I want to thank [Sadeep Jayasumana](https://github.com/sadeepj) for his excellent work with [CRF-RNN](https://github.com/sadeepj/crfasrnn_keras) post-processing unit implementation in keras. 

