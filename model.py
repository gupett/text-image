import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda, Embedding, BatchNormalization, LSTM
from keras.applications.vgg19 import VGG19
from keras import backend
from keras.optimizers import Adam
import tensorflow as tf

WORD_EMBEDDING_SIZE = 50

class triplet_loss_embedding_graph:

    def __init__(self, vocab_size, embedding_weights):

        # Could be passed right to the text embedding models, save memory?
        self.vocab_size = vocab_size
        self.word_embedding_weights = embedding_weights

        # The embedding models for images and text
        self.img_embedding_model = self.create_image_embedding_model()
        self.text_embedding_model = self.create_text_embedding_model()

        # Combined model of the embedding models, optimized based on bidirectional triplet loss
        self.model = self.create_embedding_model()
        # Compile combined model
        self.model.compile(loss='mean_absolute_error', optimizer=Adam())

    def create_embedding_model(self):
        # Initialize the text and image encoding models
        img_embedding_model = self.img_embedding_model
        text_embedding_model = self.text_embedding_model

        # Define input sizes
        image_shape = (255, 255, 3)
        text_shape = (25,)

        # Define all inputs, Image anchor and positive/ negative text
        anchor_image_example = Input(shape=image_shape, name='anchor_img')
        positive_image_example = Input(shape=image_shape, name='pos_img')
        negative_image_example = Input(shape=image_shape, name='neg_img')

        # Text anchor and positive/ negative images
        anchor_text_example = Input(shape=text_shape, name='anchor_text')
        positive_text_example = Input(shape=text_shape, name='pos_text')
        negative_text_example = Input(shape=text_shape, name='neg_text')

        # Define the triplet loss model with all the six different inputs
        anchor_image_embedding = img_embedding_model(anchor_image_example)
        positive_text_embedding = text_embedding_model(positive_text_example)
        negative_text_embedding = text_embedding_model(negative_text_example)

        anchor_text_embedding = text_embedding_model(anchor_text_example)
        positive_image_embedding = img_embedding_model(positive_image_example)
        negative_image_embedding = img_embedding_model(negative_image_example)

        # Triplet loss layer
        triplet_layer = Lambda(self.bi_directional_triplet_loss, output_shape=(1,), name='triplet_loss')
        loss = triplet_layer(
            [anchor_image_embedding, positive_text_embedding, negative_text_embedding, anchor_text_embedding,
             positive_image_embedding, negative_image_embedding])

        # Create the final model
        model = Model(inputs=[anchor_image_example, positive_text_example, negative_text_example, anchor_text_example,
                              positive_image_example, negative_image_example], outputs=loss)

        return model

    def bi_directional_triplet_loss(self, input):
        anchor_image, positive_text, negative_text, anchor_text, positive_image, negative_image = input

        # Relative importance term
        lambda1 = 1
        lambda2 = 1.5
        # The triplet margin parameter
        M = 0.05

        # Calculate euclidean distance between anchor image and positive text sample
        pos_dist_img = tf.square(tf.subtract(anchor_image, positive_text))
        pos_dist_img = tf.reduce_sum(pos_dist_img, 1) # sum over all columns (dim 1)

        # Calculate euclidean distance between anchor image and negative text sample
        neg_dist_img = tf.square(tf.subtract(anchor_image, negative_text))
        neg_dist_img = tf.reduce_sum(neg_dist_img, 1) # sum over all columns (dim 1)

        # Calculate the loss over the image anchor
        basic_img_loss = tf.add(tf.subtract(pos_dist_img, neg_dist_img), M) # calculate the loss for each image-text triplet in the batch
        img_loss = tf.reduce_mean(tf.maximum(basic_img_loss, 0.0), 0) # calculate the mean loss over the image text triplet for the batch

        # Calculate the euclidean distance from anchor text to positive image sample
        pos_dist_text = tf.square(tf.subtract(anchor_text, positive_image))
        pos_dist_text = tf.reduce_sum(pos_dist_text, 1) # sum over all columns (dim 1)

        # Calculate the euclidean distance from anchor text to positive image sample
        neg_dist_text = tf.square(tf.subtract(anchor_text, negative_image))
        neg_dist_text = tf.reduce_sum(neg_dist_text, 1) # sum over all columns (dim 1)

        # Calculate the loss over the text anchor
        basic_text_loss = tf.add(tf.subtract(pos_dist_text, neg_dist_text), M) # calculate loss for each playground triplet in the batch
        text_loss = tf.reduce_mean(tf.maximum(basic_text_loss, 0.0), 0) # calculate the mean loss over the playground triplet for the batch

        batch_loss = tf.add(tf.scalar_mul(lambda1, img_loss), tf.scalar_mul(lambda2, text_loss))

        return batch_loss

    def create_image_embedding_model(self):
        # Create VGG19-net from pre trained model, trained on imagenet, exclude the head of the model
        base_image_encoder = VGG19(weights='imagenet', include_top=False)
        image_input = base_image_encoder.input
        base_image_encoder = base_image_encoder.output

        base_image_encoder = GlobalAveragePooling2D(name='start_embedding_layers')(base_image_encoder)
        # Adding the embedding network on base image encoder
        dense_i1 = Dense(2048, activation='relu')(base_image_encoder)
        dense_i2 = Dense(512)(dense_i1)
        # Batch normalize and L2 normalize output
        batch_norm = BatchNormalization()(dense_i2)
        l2_norm_i = Lambda(lambda x: backend.l2_normalize(x, axis=1))(batch_norm)

        img_embedding_model = Model(inputs=[image_input], outputs=[l2_norm_i], name='img_emb_model')

        # Freeze all the pre trained image-encoding layers (the vgg19 layers)
        for i, layer in enumerate(img_embedding_model.layers):
            if layer.name == 'start_embedding_layers':
                break
            layer.trainable = False

        img_embedding_model.summary()

        return img_embedding_model

    def create_text_embedding_model(self):
        base_text_encoder, text_input = self.create_lstm_query_embedding_layers()

        # Adding the embedding network for the base text encoder
        dense_t1 = Dense(1024, activation='relu')(base_text_encoder)
        dense_t2 = Dense(512)(dense_t1)
        # Batch normalize and L2 normalize output
        batch_norm = BatchNormalization()(dense_t2)
        l2_norm_t = Lambda(lambda x: backend.l2_normalize(x, axis=1))(batch_norm)

        text_embedding_model = Model(inputs=[text_input], outputs=[l2_norm_t], name='text_emg_model')
        text_embedding_model.summary()

        return text_embedding_model

    def create_query_embedding_layers(self):

        text_input = Input(shape=(None,))

        word_embedding = Embedding(input_dim=self.vocab_size, #embedding_matrix.shape[0], # Vocabulary size
                                output_dim=WORD_EMBEDDING_SIZE, #embedding_matrix[1], # vector encoding size
                                weights=[self.word_embedding_weights], # Pre trained word embedding
                                input_length= 25,
                                trainable=False, # If weights can be modified through back propagation
                                mask_zero=True, # klipp away the zero embeddings
                                name='word_embedding')(text_input)

        query_embedding = Lambda(lambda w_emb: tf.reduce_mean(w_emb, axis=1), name='mean_of_word')(word_embedding) # calculate the mean word vector

        return query_embedding, text_input

    def create_lstm_query_embedding_layers(self):

        text_input = Input(shape=(None,))

        word_embedding = Embedding(input_dim=self.vocab_size, #embedding_matrix.shape[0], # Vocabulary size
                                output_dim=WORD_EMBEDDING_SIZE, #embedding_matrix[1], # vector encoding size
                                weights=[self.word_embedding_weights], # Pre trained word embedding
                                input_length= 25,
                                trainable=False, # If weights can be modified through back propagation
                                mask_zero=True, # klipp away the zero embeddings
                                name='word_embedding')(text_input)


        lstm_1 = LSTM(512)(word_embedding)
        return lstm_1, text_input

if __name__ == '__main__':
    model_class = triplet_loss_embedding_graph()
    model = model_class.model
    model.summary()