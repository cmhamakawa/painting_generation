# Painting Generation with GANs
UCLA PIC 16B Project to generate paintings via generative adversarial networks (GANs).

## Introduction
This project aimed to generate paintings via deep learning models and compare their results,  particularly generative adversarial networks (GANs). Through this project, we gained practical experience developing GANs and their variants in code, leveraged image transformation and pre-processing techniques, and learned how to utilize Streamlit to create a user-friendly interactive frontend for the model(s). 


## How to Run Demo:
Pull the git repo and cd into the rpo.
Run the following command: streamlit run streamlit_demo.py
You will then be able to interact with the dataset and our DCGAN models (both the discriminator and the generator)!

## Organization of Repo
The directories ‘can’ and ‘dcgan’ contain code for their models respectively. The directories ‘Dcgan_models’ and ‘data’ is used by streamlit_demo.py. Everything in the main directory not in a subdirectory is also used directly or indirectly by the streamlit_demo.py file, which is the main source of the demo code.

## Relevant Concepts
### Generative Models
Generative models capture the distribution of the data and generate new data instances. They include the distribution of the data itself. For instance, a generative model can generate new images of houses, cats, or really anything.
### Generative Adversarial Network (GAN)
We introduce our foundational model, a deep-learning based generative model known as GAN. GANs stand for generative adversarial networks. GANs are ultimately a great way of creating a generative model: using two (sub) neural networks. These two neural networks ultimately work together to make the generative model. We now introduce the names of these two neural networks:
The generator: A neural network that produces synthetic or fake images that look like real training images
The discriminator: A neural network that checks if (or classifies) the image as real or fake
In essence, the generator produces the fake image and tries to fool the discriminator into thinking the image is real. In return, the discriminator tries to classify the image as fake or real correctly. Both neural networks try to minimize their loss in such a way that they improve together. We include the diagram below for a more visual understanding of what we have discussed.
### Deep Convolutional GAN (DCGAN)
DCGAN stands for deep convolutional generative adversarial network. At its core, it is a GAN model but with a slight variation. The discriminators of both a GAN and DCGAN model are ultimately the same; the difference lies in the generator of the two models. The GAN generator uses a fully connected neural network, while the DCGAN generator uses deep convolutional nets. Particularly, the DCGAN generator uses transposed convolutional layers. Rather than downsampling as in a traditional convolutional layer, a transposed convolutional layer upsamples images, or increases the magnitude of its current dimensions using pre-existing values. 

### CAN
A CAN model functions very similarly to the DCGAN model but with an additional aspect. The way a traditional GAN is designed, as it trains, the generator will improve at producing images which closely resemble the training data, but in certain contexts, such as visual art, this may be undesirable for a number of reasons, from potential plagiarism to simply a lack of originality. Thus, CANs aim to increase the creativity of the generator by providing it additional information on the style of the art. Specifically, in the case of a CAN, the discriminator not only predicts whether a given image is real or AI-generated, it also tries to classify it into one of the dataset’s known styles. In the case of the generated image, this style ambiguity loss is received as a second input to the generator, which now has to generate images that look like the paintings in the dataset but also cannot be too similar to already existing styles (it is penalized if so). These two contradictory forces thus push the model to create novel art.


