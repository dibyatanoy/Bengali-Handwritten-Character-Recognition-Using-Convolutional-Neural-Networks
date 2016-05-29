# This is the image processing module, which I used to preprocess my images, augment my dataset, and
# organize them into a structure suitable for input to a machine learning model

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
from six.moves import range

# filenames for the training and testing folders
train_folder = "Train"
test_folder = "Test"

# standard dimensions to which all images will be rescaled
dimensions = (50, 50)

# maximum angle by which the image can be rotated during data augmentation
max_angle = 15

# function to rotate an image by a given angle and fill in the black corners created
# with a specified color
def rotate_img(image, angle, color, filter = Image.NEAREST):

    if image.mode == "P" or filter == Image.NEAREST:
        matte = Image.new("1", image.size, 1) # mask
    else:
        matte = Image.new("L", image.size, 255) # true matte
    bg = Image.new(image.mode, image.size, color)
    bg.paste(
        image.rotate(angle, filter),
        matte.rotate(angle, filter)
    )
    return bg


# function to turn grey-colored backgrounds to white. r, b and g specify the
# exact shade of grey color to eliminate. Source: stackoverflow.
def make_greyscale_white_bg(im, r, b, g):

    im = im.convert('RGBA')   # Convert to RGBA


    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace grey with white... (leaves alpha values alone...)
    grey_areas = (red == r) & (blue == b) & (green == g)
    data[..., :-1][grey_areas.T] = (255, 255, 255) # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')   # convert to greyscale image


    return im2


# Make a specified number of copies if the given image by rotating the original image by
# some random angle, and save the images according to the naming scheme followed by the original images

def random_rotate(img, copies, curr_filename, path):

    c_color = img.getpixel((0,0))       # get the pixel values of top-left corner of image

    for i in range(copies):

        # rotate image by a random angle from [-max_angle, max_angle], using the c_color to fill in the corners
        new_im = rotate_img(img, np.random.randint((0 - max_angle), max_angle), c_color)
        # save new image to file
        new_im.save(os.path.join(path, "bcc" + str(curr_filename).zfill(6) + ".bmp"))

        curr_filename = curr_filename + 1



# augment the dataset by adding random rotations. The count of the original images is needed
# for naming the new images in a sequential order
def augment_by_rotations(folder, prev_cnt):

    classes = [os.path.join(folder, d) for d in sorted(os.listdir(folder))]  # get list of all sub-folders in folder

    for path_to_folder in classes:

        if os.path.isdir(path_to_folder):
            images = [os.path.join(path_to_folder, i) for i in sorted(os.listdir(path_to_folder)) if i != '.DS_Store']
            filename = prev_cnt
            for image in images:

                im = Image.open(image)

                # make 4 copies of each image, with random rotations added in
                random_rotate(im, 4, filename, path_to_folder)
                filename = filename + 4

            print("Finished augmenting " + path_to_folder)


# function to invert colors (black -> white and white-> black). Since most of the image consists
# of white areas, specified by (255, 255, 255) in RGB, inverting the colors means more zeros, making
# future operations less computationally expensive

def invert_colors(im):

    im = im.convert('RGBA')   # Convert to RGBA
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability


    # Replace black with red temporarily... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = (255, 0, 0) # Transpose back needed

    # Replace white areas with black
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    data[..., :-1][white_areas.T] = (0, 0, 0) # Transpose back needed

    # Replace red areas (originally white) with black
    red_areas = (red == 255) & (blue == 0) & (green == 0)
    data[..., :-1][red_areas.T] = (255, 255, 255) # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')   # convert to greyscale image


    return im2


# function to test the other functions on a specified image
# this is not needed once the other functins are confirmed to be working
def test_rotations():

    img = Image.open("Train/172/bcc000002.bmp")

    #img = img.rotate(30)

    img = img.resize(dimensions)



    rot = make_greyscale_white_bg(img, 127, 127, 127)

    rot = invert_colors(rot)
    c_color = rot.getpixel((0, 0))
    rot = rotate_img(rot, 10, c_color)

    w, h = rot.size
    rot.show()


# function to process images (resizing, removal of grey backgrounds if any, color inversion, greyscale conversion)

def process_images(folder):

    classes = [os.path.join(folder, d) for d in sorted(os.listdir(folder))]  # get list of all sub-folders in folder
    img_cnt = 0

    for class_x in classes:

        if os.path.isdir(class_x):

            # get paths to all the images in this folder
            images = [os.path.join(class_x, i) for i in sorted(os.listdir(class_x)) if i != '.DS_Store']

            for image in images:

                img_cnt = img_cnt + 1

                if(img_cnt % 1000 == 0):                # show progress
                    print("Processed %s images" % str(img_cnt))

                im = Image.open(image)
                im = im.resize(dimensions)   # resize image according to dimensions set

                im = make_greyscale_white_bg(im, 127, 127, 127) # turn grey background (if any) to white, and
                                                                  # convert into greyscale image with 1 channel

                im = invert_colors(im)
                im.save(image)   # overwrite previous image file with new image

    print("Finished processing images, images found = ")
    print(img_cnt)

process_images(test_folder)
process_images(train_folder)

augment_by_rotations(train_folder, 240)


# The code below organizes the processed images into structures suitable for use with ML models
# A lot of the code is obtained from the assignments in the Google Deep Learning course in Udacity

image_size = 50  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


# function to load all images from given folder, then convert the dataset into a 3D array (image index, x, y)
#  of floating point values, normalized to have approximately zero mean and
#  standard deviation ~0.5 to make training easier.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""

  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)

  num_images = 0
  for image_index, image in enumerate(image_files):
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) -      # normalize data
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.') # skip unreadable files

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:                                   # check if a given min. no. of images
    raise Exception('Many fewer images than expected: %d < %d' %    # has been loaded
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset


# function to store the normalized tensors obtained from the load_letter function in
# .pickle files for later use

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  folders_list = os.listdir(data_folders)
  for folder in folders_list:

    #print(os.path.join(data_folders, folder))
    curr_folder_path = os.path.join(data_folders, folder)
    if os.path.isdir(curr_folder_path):
        set_filename = curr_folder_path + '.pickle'
        dataset_names.append(set_filename)
        if os.path.exists(set_filename) and not force:
          # You may override by setting force=True.
          print('%s already present - Skipping pickling.' % set_filename)
        else:
          print('Pickling %s.' % set_filename)
          dataset = load_letter(curr_folder_path, min_num_images_per_class) # load and normalize the data
          try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
                f.close()
          except Exception as e:
            print('Unable to save data to', set_filename, ':', e)

  return dataset_names

train_datasets = maybe_pickle(train_folder, 1050, True)     # load, normalize and pickle the train and test datasets
test_datasets = maybe_pickle(test_folder, 58, True)



# function to make two empty arrays, one for the input data and one for the labels

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

# function to merge all the images in the given pickle file. Part of the training dataset is used to
# create a validation dataset for hyperparameter tuning.

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        f.close()
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

# set the no. of images to be used in each dataset
train_size = 50000
valid_size = 5000
test_size = 3000


valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


# shuffle the images in each dataset randomly, and their corresponding labels

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


# file name for saving all the processed datasets and their corresponding label tensors
pickle_file = 'bengaliOCR.pickle'


# save all the processed datasets into one large pickle file for later usage
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

