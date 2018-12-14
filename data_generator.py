from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input
import pickle
import random
import numpy as np

'''
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
'''

IMAGE_EXTENSION = 'Data/Flicker8k_Dataset/'
TEXT_EXTENSION = 'Data/Flickr8k_text/'

TRAINING_FILE = 'training_very_small.txt'

IMG_HEIGHT = 255
IMG_WIDTH = 255


class Data_generator:

    def __init__(self, batch_size, pre_trained=False):
        self.images, self.annotations = self.load_annotations()

        self.pre_trained = pre_trained

        if self.pre_trained:
            # Load the tokenizer from file
            with open('./model/tokenizer/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self.create_tokenizer()

        # +2 since zero padding is used in the embedding layer
        self.vocabulary_size = len(self.tokenizer.word_index) + 2

        self.batch_size = batch_size
        self.batch_nr = 0

        # Number of batches for each epoch
        self.batch_per_epoch = self.batches_per_epoch()
        #print('The number of batches per epoch is: {}'.format(self.batch_per_epoch))

    def batches_per_epoch(self):
        return (len(self.images) - 1) // self.batch_size

    def load_annotations(self):
        with open(TEXT_EXTENSION + TRAINING_FILE) as file:
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

####### PREE PROCESSING ################

    def create_tokenizer(self):
        text = ''
        for row in self.annotations:
            for r in row:
                text += r

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])

        return tokenizer

    def get_embedding_matrix(self):

        # Load existing embedding matrix or create a new one
        if self.pre_trained:
            embedding_matrix = self.load_embedding_matrix()
        else:
            # load the entire embedding from file into a dictionary
            embeddings_index = dict()
            f = open('Data/glove/glove.6B.300d.txt', encoding='utf-8')
            for line in f:
                # splits on spaces
                values = line.split()
                # the word for the vector is the first word on the row
                word = values[0]
                # Extra the vector corresponding to the word
                vector = np.asarray(values[1:], dtype='float32')
                # Add word (key) and vector (value) to dictionary
                embeddings_index[word] = vector
            f.close()

            # Initialize an embedding matrix with shape vocab_size x word_vector_size
            embedding_matrix = np.zeros((self.vocabulary_size, 300))
            # Go through the tokenizer and for each index add the corresponding word vector to the row
            for word, index in self.tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

            # Save the new embedding matrix
            self.save_embedding_matrix(embedding_matrix)

        return embedding_matrix

    def save_embedding_matrix(self, embedding_matrix):
        np.save('./model/Embedding_weights/embedding_matrix.npy', embedding_matrix)

    def load_embedding_matrix(self):
        return np.load('./model/Embedding_weights/embedding_matrix.npy')

####### PREE PROCESSING ENDS ################

    def new_epoch(self):
        # Shuffle the annotations and images
        combined = list(zip(self.images, self.annotations))
        random.shuffle(combined)

        self.images[:], self.annotations[:] = zip(*combined)

    def load_image_batch(self, image_names):
        images = np.zeros((self.batch_size, IMG_HEIGHT, IMG_WIDTH, 3))
        for i, image_name in enumerate(image_names):
            image = load_img(IMAGE_EXTENSION + image_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
            images[i, :, :, :] = image

        # do the normalization required for vgg19
        images = preprocess_input(images)
        return images

    def batch_generator(self):
        # batches for the following inputs are created: anchor_img, positive_annotation, negative_annotation,
        # anchor_text, positive_img, negative_img
        while True:

            # Check if it is time for new epoch, in such case shuffle the data and zero the batch number
            if self.batch_nr == self.batch_per_epoch:
                self.new_epoch()
                self.batch_nr = 0

            # Creation of the image anchor inputs for the batch
            # All the image anchor names for the batch are put into a list
            anchor_img_name_batch = self.images[self.batch_nr*self.batch_size: (self.batch_nr+1)*self.batch_size]
            # The images from the list are loaded in to np array
            anchor_img_batch = self.load_image_batch(anchor_img_name_batch)


            positive_annotation_batch = list()
            for i in range(self.batch_nr * self.batch_size, (self.batch_nr + 1) * self.batch_size):
                annotation_sample = random.randint(0, 4)

                # Get an encoding of the annotation
                annotation_tokenized = self.tokenizer.texts_to_sequences([self.annotations[i][annotation_sample]])[0]

                padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
                positive_annotation_batch.append(padded_annotation_tokenized)

            positive_annotation_batch = np.array(positive_annotation_batch)
            positive_annotation_batch = np.squeeze(positive_annotation_batch)


            annotation_sample = random.randint(0, 4)
            annotation_tokenized = self.tokenizer.texts_to_sequences([self.annotations[(self.batch_nr + 1) * self.batch_size][annotation_sample]])[0]
            padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
            negative_annotation_batch = [padded_annotation_tokenized]
            for i in range((self.batch_nr * self.batch_size), ((self.batch_nr + 1) * self.batch_size) - 1):
                annotation_sample = random.randint(0, 4)

                # Get an encoding of the annotation
                annotation_tokenized = self.tokenizer.texts_to_sequences([self.annotations[i][annotation_sample]])[0]
                padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
                negative_annotation_batch.append(padded_annotation_tokenized)

            negative_annotation_batch = np.array(negative_annotation_batch)
            negative_annotation_batch = np.squeeze(negative_annotation_batch)

            anchor_annotation = list()
            for i in range(self.batch_nr * self.batch_size, (self.batch_nr + 1) * self.batch_size):
                annotation_sample = random.randint(0, 4)
                anchor_annotation.append(self.annotations[i][annotation_sample])

            positive_img_name_batch = anchor_img_name_batch.copy()

            negative_img_name_batch = [self.images[(self.batch_nr+1)*self.batch_size]]
            negative_img_name_batch.extend(self.images[self.batch_nr*self.batch_size: ((self.batch_nr+1)*self.batch_size - 1)])

            # Shuffle the anchor_annotation, positive_image_name and negative_image_name together
            combined = list(zip(anchor_annotation, positive_img_name_batch, negative_img_name_batch))
            random.shuffle(combined)
            anchor_annotation, positive_img_name_batch, negative_img_name_batch = zip(*combined)

            # turn the anchor annotations in to one hot sequences
            anchor_annotation_batch = list()
            for annotation in anchor_annotation:
                annotation_tokenized = self.tokenizer.texts_to_sequences(annotation)[0]
                padded_annotation_tokenized = pad_sequences([annotation_tokenized], maxlen=25)
                anchor_annotation_batch.append(padded_annotation_tokenized)

            anchor_annotation_batch = np.array(anchor_annotation_batch)
            anchor_annotation_batch = np.squeeze(anchor_annotation_batch)

            positive_img_batch = self.load_image_batch(positive_img_name_batch)
            negative_img_batch = self.load_image_batch(negative_img_name_batch)

            # Increase batch number
            self.batch_nr += 1

            yield {'anchor_img': anchor_img_batch, 'pos_text': positive_annotation_batch,
                   'neg_text': negative_annotation_batch, 'anchor_text': anchor_annotation_batch,
                   'pos_img': positive_img_batch, 'neg_img': negative_img_batch}, np.zeros(self.batch_size)

if __name__ == '__main__':

    load = Data_generator(batch_size=5)
    load.new_epoch()

    '''
    token = load.tokenizer
    print(token.texts_to_sequences(['blsad is good at'])[0])
    print(len(token.index_word))
    print(token.index_word[296])
    embedding = load.get_embedding_matrix()
    print(embedding[0,:])
    print(token.index_word[1])
    print(embedding[1, :])

    print(token.index_word[296])
    print(embedding[296, :])
    '''



    gen = load.batch_generator()
    t, y = next(gen)

    print(t['pos_text'].shape)
    print(t['neg_text'].shape)
    print(t['anchor_img'].shape)

    print(t['anchor_text'].shape)
    print(t['pos_img'].shape)
    print(t['neg_img'].shape)


    '''
    for i in range (0):
        print(t['pos_text'][i])
        print(t['neg_text'][i])
        plt.imshow(t['anchor_img'][i]/255.0)
        plt.show()

    for i in range(5):
        print(t['anchor_text'][i])
        plt.subplot(221)
        plt.imshow(t['pos_img'][i]/255.0)
        plt.subplot(222)
        plt.imshow(t['neg_img'][i] / 255.0)
        plt.show()
    '''



    '''
    for i, row in enumerate(load.annotations):
        value = load.images[i]
        print('Key: {}, Value: {}'.format(row, value))
        if i == 10:
            break

    print('Key: {}, Value: {}'.format(load.annotations[-1], load.images[-1]))
    '''