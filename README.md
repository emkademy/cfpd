# **C**onvolutional **F**acial **P**arts **D**etector - CFPD
This repository is implementation of a facial parts detector
that is based on a moderate size Convolutional Neural Network.
Given a face image, the model can detected eyes (together with
eyebrows), nose and mouth. It can run on top of any face 
detector. Since the network is not very large, the performance
is in real-time.


### Prerequisites

---
(***Note:*** There is a docker image for this repo. If it is
better for you, please see below.)
* Python 3.6
* Pytorch 0.4.1
* OpenCV 4.0.0.21
* Matplotlib 3.0.2
* Dlib 19.4.0 (For camera demo)

The model was trained and tested using GPU, thus if possible
use GPU for training/testing. I included *'requirements.txt'*
file, if you want you can use it to create the same environment
that I have. To do that:

```bash
conda create --name <ENV_NAME> --file requirements.txt
conda activate <ENV_NAME>
```
Note that this may take a while. If you would like to install
the dependencies via pip, you can go ahead and:

```bash
pip install opencv-python
pip install dlib
pip install matplotlib
```
And for pytorch, please visit their [webpage](https://pytorch.org/get-started/locally/) for installation instructions.


### Camera Demo

---
![](output.gif)

If you are not interested in training/testing and just want
to see what the model can do, then the script you are looking
for is **camera_demo.py**. Like its name suggests, it is a
demo that uses your camera. Since this model works on top of
a face detector, I implemented this demo using *dlib* library's
face detector. So in order to use the demo, you will need that.

*To download the pretrained model:*
```bash
# On Linux or Mac
sudo chmod u+x download_models.sh
./download_models.sh
```
On windows you will need to donwload the models manually from [here](https://1drv.ms/f/s!AjmsnJ7wS5KLgoVAyAaiR6HI2_YQkA)
(downlaod all **data** folder and save it to this directory)
 
*To run the demo:*
```python
python camera_demo.py
```

### Training

---
The model was trained using the 300W, LFPW, HELEN, AFW and 
IBUG datasets ([download link](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)) 
If you want to repeat the training process, download these
datasets. If you want to use the default setup, extract them 
into *./data/images/DATASET_NAME_AS_WRITTEN_ABOVE*. Then run:

```bash
python main.py --section-name cfpd
```

If you want to use your own setup, or a different dataset, 
you will need to modify ***./config.ini*** file and ***./constants.py***
file. In *./config.ini* file we basically have the configuration
parameters such as: parameters for training and parameters for
datasets. And ***./constants.py*** file parses the config file
and saves all of the parameters defined in a class for convenience
and to not have 1 million parameter definitions all around the code.

As it was said 300-W dataset was used to train this model.
So, as well as images, ***.pts (facial landmarks)*** files
are necessary to train the model.

A little explanation for the ***n_augmented_images*** parameter
in *./config.ini* file can be necessary;

* *n_augmented_images = 0* (no data augmentation)
* *n_augmented_images = 1* (images will be normalized to canonical pose) 
* *n_augmented_images > 1* (data augmentation by randomly scaling, rotating, and transforming)


### Testing

---
All pretrained models can be tested using ***./tests.py***
script:

```bash
python tests.py --section-name cfpd
```
### Docker image

---
Probably quickest way to start using this repo:

[Docker image](https://hub.docker.com/r/kivancyuksel/alluneed)

However please keep in mind that if you want to use the docker
image for camera demo, you will need to make
your display available in the docker image. 

Also, during training you might get an error if you don't have
your display available in the docker image. The reason for that
is because during training the loss graph is drawn using matplotlib.
If you get any error related to display during training, just
comment this line out in *trainer_and_tester.py*:

```python
track_losses(losses, model_save_path)
```

### Licence

---
While the code is licensed under the MIT license, 
please remember that the pretrained models that are provided
with this repository are trained on 300-W dataset. This
dataset can be used only for research purposes. For details
please refer to this [link](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
