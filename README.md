#Bengali Handwritten Character Recognition using Convolutional Neural Networks (CNNs)
Handwritten character recognition is an old problem that has been extensively studied for many different languages around the world. Although very high recognition rate has been achieved repeatedly using a variety of methods on the English character set, there exists limited work on the Bengali character set, and most of these achieved recognition accuracies below 90%. Here, I implement a number of different convolutional neural networks (CNNs) to this problem. To the best of my knowledge, there exists only 1 paper by *Akhand et al. (2015)* that has employed CNNs to the bengali character set, and that, too, achieved 85.96% accuracy.

Some facts about the Bengali language:

- Bengali is the native language of Bangladesh, and is spoken by 300 million people worldwide, making it the seventh most spoken language in the world by total number of native speakers. 
- The Bengali character set consists of 50 basic characters - 11 vowels and 39 consonants.
- Characters can also be compounded to form many different combinations (~300). Here, I work with the recognition of the 50 basic characters only.

A lot of the trouble with Bengali character recognition stems from the fact that there exist a number of character pairs that are almost identical, differing only by a dash or a dot (for instance, ড and ড়). Feature-extraction based methods have worked decently well, but only one such model, in the work of *Bhattacharya et al. (2012)* has exceeded the 90% recognition rate.

###Dataset

The dataset was obtained online from the [CMATERdb](https://www.dropbox.com/s/55bhfr3ycvsewsi/CMATERdb%203.1.2.rar) pattern recognition database repository. It consists of a Train folder and a Test folder, containing 12,000 and 3,000 images respectively.

###Pre-processing

All the images were rescaled to standard dimensions of 50x50. The gray backgrounds in a lot of the images were replaced with white backgrounds, and then colors inverted (black to white, and white to black) to reduce computational cost. Finally, all images were converted into grayscale format.

The training dataset was augmented by making 4 copies of each image with small random rotations (between -15 and 15 degrees) added in, and unwanted black corners fixed. The images in the train folder (about 60,000 after pre-processing, minus some very noisy images that I hand-picked and deleted) were divided into two: a train dataset and a validation dataset for hyper-parameter training.

The final train datasets used had 50,000 images (1000 images per class, for 50 classes), while the validation dataset had 5000. A sample of the images after processing is shown:


![Sample images](https://github.com/dibyatanoy/Bengali-Handwritten-Character-Recognition-Using-Convolutional-Neural-Networks/blob/master/result_screenshots/processed_sample.png)


###Models Used

I implemented a number of different neural net architectures, including:
- Simple (non-convolutional) neural network with 2 hidden ReLU layers
- Network with 2 convolutional layers and 1 fully connected layer
- Network with 2 convolutional layers, max-pooling layers after each and 2 fully connected layers at the end
- Network with 2 convolutional layers, max-pooling after each, an inception module and 2 fully connected layers at the end

I have included the python code for each model, together with the actual iPython (Jupyter) notebooks on which I carried out my experiments. I used Google’s Tensorflow library to implement these networks.

###Results

The best result was obtained using the final network architecture which includes an inception module. The test set accuracy was at **94.2%** and training set accuracy at **98.9%**.

![alt text](https://github.com/dibyatanoy/Bengali-Handwritten-Character-Recognition-Using-Convolutional-Neural-Networks/blob/master/result_screenshots/Conv_nets_inception.png)