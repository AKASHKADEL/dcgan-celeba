# DCGAN 

This is my second implementation of dcgan. My first implementation was on mnist dataset which could be found [here](https://github.com/AKASHKADEL/dcgan-mnist). This implementation is on the celebrity faces dataset. This is the [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to the dataset.

# Introduction

Deep Convolutional GAN is one of the most coolest and popular deep learning technique. It is a great improvement upon the [original GAN network](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) that was first introduced by Ian Goodfellow at NIPS 2014. (DCGANs are much more stable than Vanilla GANs) DCGAN uses the same framework of generator and discriminator. This is analogous to solving a two player minimax game: Ideally the goal of the discriminator is to be very sharp in distinguishing between the real and fake data, whereas, generator aims at faking data in such a way that it becomes nearly impossible for the discriminator to classify it as a fake. The below gif shows how quickly dcgan learns the distribution of celebrity images and generates real looking people. The gif is created for both, a fixed noise and variable noise:-

<p float="left">
  <img src="https://github.com/AKASHKADEL/dcgan-celeba/blob/master/results/variable_noise/animated.gif" width="400" height="400" />
  <img src="https://github.com/AKASHKADEL/dcgan-celeba/blob/master/results/fixed_noise/animated.gif" width="400" height="400" />
</p>

# Quick Start

To get started and to replicate the above result, follow the instructions in this section. This wil allow you to train the model from scratch and help produce basic visualizations. 

## Dependencies:

* Python 3+ distribution
* PyTorch >= 1.0

Optional:

* Matplolib and Imageio to produce basic visualizations.
* Cuda >= 10.0

## Dataset:
The dataset was downloaded from this [link](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Once you have downloaded the images, create a ```train```folder. This folder should contain the celebA folder which in turn contains the celebrity images. (I have added 10 sample images in train folder just for reference. Obviously, I cannot add the entire dataset due to memory limit.) We will make use [torchvision's imagefolder](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder) library to directly read the images from that folder. This makes reading, normalizing and cropping images very easy. The following method reads the images from ``` train/celeb-a/ ``` folder and creates a dataloader with a given batch size:

```     
def get_data_loader(root, batch_size):
    
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.ImageFolder(root=root, transform=transform)

    # Data Loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
```

## Steps:
Once everything is installed, you can go ahead and run the below command to train a model on 100 Epochs and store the sample outputs from generator in the ```results``` folder.

```python main.py --num-epochs 100 --output-path ./results/ ```

You can also generate sample output using a fixed noise vector (It's easier to interpret the output on a fixed noise. Ex: the above gif), use this

```python main.py --num-epochs 100 --output-path ./results/ --use-fixed ```

You can change the model setting by playing with the learning rate, num_epochs, batch size, etc

## Outputs

The above code will store 100 images in the folder ```./results/fixed_noise```, each storing the output after every epoch. Also, the imageio library will then take these 100 images a create a gif out of it with fps=5. The final gif will be stored in the same folder. ie., ```./results/fixed_noise/animated.gif```

# References:

[1] https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf <br>
[2] https://arxiv.org/pdf/1511.06434.pdf <br>
[3] https://github.com/soumith/ganhacks <br>
[4] https://medium.com/activating-robotic-minds/up-sampling-with-transposed-convolution-9ae4f2df52d0 <br>




