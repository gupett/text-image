from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from time import time
import pickle

from model import triplet_loss_embedding_graph
from data_generator import Data_generator
from validation_generator import validation_generator
from evaluate_model import evaluate_model

BATCH_SIZE = 5
EPOCHS = 1
PRE_TRAINED = False

class training:

    def __init__(self):

        # Create the batch data generator and get otger required information of the data
        self.training_generator = Data_generator(BATCH_SIZE)
        self.tokenizer = self.training_generator.tokenizer
        self.vocab_size = self.training_generator.vocabulary_size
        # The weights (glove) for the word embedding matrix used in model
        embedding_weights = self.training_generator.get_embedding_matrix()

        # Validation data generator
        self.validation_generator = validation_generator(self.tokenizer, BATCH_SIZE)

        # The playground embedding class
        self.TLEG = triplet_loss_embedding_graph(vocab_size=self.vocab_size, embedding_weights=embedding_weights)
        # The playground embedding triplet loss model
        self.model = self.TLEG.model
        if PRE_TRAINED:
            # Load pre-trained weights
            self.model.load_weights('./model/best_weights.hdf5')
        self.model.summary()

    # Function defining how the model should be trained
    def train(self, epochs):
        # Save the tokenizer to file for use at inference time
        with open('./model/tokenizer/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # define a tensorboard callback
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        # define early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=0, mode='auto')

        # Save the best model
        file_path = './model/best_weights.hdf5'
        checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
        callbacks = [checkpoint, earlystop, tensorboard]

        self.model.fit_generator(generator=self.training_generator.batch_generator(),
                                 steps_per_epoch=self.training_generator.batch_per_epoch,
                                 epochs=epochs, verbose=1, callbacks=callbacks,
                                 validation_data=self.validation_generator.batch_generator(),
                                 validation_steps=self.validation_generator.batch_per_epoch
                                 )


        self.model.save_weights('./model/last_weights.hdf5')

        # Save the text and image embedding weights
        self.TLEG.img_embedding_model.save_weights('./model/last_img_embedding_weights.hdf5')
        self.TLEG.text_embedding_model.save_weights('./model/last_text_embedding_weights.hdf5')

        # Evaluate the trained model based on recall 1, 5 and 10 accuracy
        evaluator = evaluate_model()
        evaluator.top_1_5_10_accuracy()

if __name__ == '__main__':
    Trainer = training()
    Trainer.train(EPOCHS)