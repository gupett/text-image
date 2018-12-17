from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.vgg19 import preprocess_input
import pickle
from scipy import spatial
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from model import triplet_loss_embedding_graph

IMAGE_EXTENSION = 'Data/Flicker8k_Dataset/'
TEXT_EXTENSION = 'Data/Flickr8k_text/'

EVALUATION_FILE = 'training_very_small.txt'
from PIL import Image

IMG_HEIGHT = 500
IMG_WIDTH = 300

class model_demo:

    def __init__(self):

        # Load the tokenizer from file
        with open('./model/tokenizer/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        self.tokenizer = tokenizer

        # Load the embedding matrix, needed for model initialization
        embedding_matrix = np.load('./model/Embedding_weights/embedding_matrix.npy')
        vocab_size = embedding_matrix.shape[0]

        # Initialize the text and image embedding model and load pre trained weights
        TLEG = triplet_loss_embedding_graph(vocab_size, embedding_matrix)
        self.image_embedding_model = TLEG.img_embedding_model
        self.image_embedding_model.load_weights('./model/last_img_embedding_weights.hdf5')
        self.text_embedding_model = TLEG.text_embedding_model
        self.text_embedding_model.load_weights('./model/last_text_embedding_weights.hdf5')

        # Load the validation data for accuracy check
        self.images, self.annotations = self.load_validation_data()

        # Initialize the query tree from where the nearest neighbours will be taken
        self.encoding_tree = self.image_encoding_tree()

    # find the k nearest images for a text query
    def query_image(self, query, k=1, verbose=1):
        # Tokenize the query input
        query = query.lower()
        annotation_tokenized = self.tokenizer.texts_to_sequences([query])[0]
        padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
        #print(padded_annotation_tokenized)

        query_embedding = self.text_embedding_model.predict(padded_annotation_tokenized)

        predictions = self.encoding_tree.query(query_embedding[0], k)
        #print(predictions)
        image_nr_prediction = predictions[1]

        if verbose == 1:
            print('The closest images are:')
            for nr in image_nr_prediction:
                print('nr: {}, with name: {}'.format(nr, self.images[int(nr)]))

            image_frame = self.concatinate_images(image_nr_prediction)

            plt.imshow(image_frame / 255.0)
            plt.show()


        return image_nr_prediction

    def encode_images(self):
        images = self.load_images()
        prediction = self.image_embedding_model.predict(images)
        #print(prediction.shape)

        #print(prediction)
        return prediction

    def image_encoding_tree(self):
        image_encoding = self.encode_images()
        # Put the image encodings in a tree structure, for better query
        encoding_tree = spatial.KDTree(image_encoding)

        return encoding_tree


    def load_images(self):
        images = np.zeros((len(self.images), IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(self.images):
            image = load_img(IMAGE_EXTENSION + image_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
            # do the normalization required for vgg19
            images[i, :, :, :] = image

        images = preprocess_input(images)
        return images


    def load_images_new(self, image_names):
        images = np.zeros((len(self.images), IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(image_names):
            image = load_img(IMAGE_EXTENSION + image_name, target_size=(IMG_HEIGHT, IMG_WIDTH))

            # do the normalization required for vgg19
            images[i, :, :, :] = image

        return images

    '''
    def load_images_new(self, image_names):
        images = np.zeros((len(self.images), IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(image_names):
            image = Image.open(IMAGE_EXTENSION + image_name)
            image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)


            #image = load_img(IMAGE_EXTENSION + image_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
            # do the normalization required for vgg19
            
        return images
    '''

    def concatinate_images(self, image_numbers):
        # Get name for image numbers
        image_names = [self.images[img_nr] for img_nr in image_numbers]

        tot_width = IMG_WIDTH * len(image_names)

        images = self.load_images_new(image_names)

        image_frame = np.zeros((IMG_HEIGHT, tot_width, 3))

        for i in range(len(image_names)):
            image_frame[:,i*IMG_WIDTH:(i+1)*IMG_WIDTH,:] = images[i,:,:,:]

        #temp = image_frame[:,:,0]

        #image_frame[:,:,0] = image_frame[:,:,2]
        #image_frame[:,:,2] = temp

        return image_frame


    def load_validation_data(self):
        # with open(TEXT_EXTENSION + 'Flickr8k.token_very_small.txt') as file:
        with open(TEXT_EXTENSION + EVALUATION_FILE) as file:
            file_content = file.readlines()

        # Annotations is a 2d list, each row is the annotations for an image and each column is a spesific annotation or the image
        annotations = list()
        image_annotations = list()
        images = list()
        # Get the name of the first image in the list
        current_img = file_content[0].split('#', 1)[0].lower()
        for line in file_content:
            line = line.lower()
            line = line.replace('\n', '')
            img, ann = line.split('\t', 1)
            img, _ = line.split('#', 1)
            if current_img == img:
                image_annotations.append(ann)
            else:
                # Append the accumulated annotations for a image and zero the list
                annotations.append(image_annotations)
                image_annotations = []
                image_annotations.append(ann)
                images.append(current_img)
                current_img = img

        # Add the last image and its annotations
        images.append(current_img)
        annotations.append(image_annotations)

        return images, annotations

if __name__ == '__main__':
    evaluator = model_demo()
    query = 'A girl going into a wooden building .'
    images = evaluator.query_image(query=query, k=10, verbose=1)
    print(images)

