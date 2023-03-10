# Basic
import numpy as np
import matplotlib.pyplot as plt
# Neural Network
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras


#---> Create class for our Neural network
class AutoE:
    def __init__(self, shape, bottelneck, filter_shape):
        self.grid_size = shape
        self.bottelneck = bottelneck
        self.filter_shape = filter_shape

    def create_model(self, summary):

        #--->>> Encoder 
        encoder_input = layers.Input(
        shape = self.grid_size, name="encoder input")

        # Convolutional Layers
        # Layer 1
        X = layers.Conv2D(4, self.filter_shape, activation="relu",
                    padding="same")(encoder_input)
        X = layers.Conv2D(4, self.filter_shape, activation="relu",
            padding="same")(X)

        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        # Layer 2
        X = layers.Conv2D(8, self.filter_shape, activation="relu",
                    padding="same")(X)
        X = layers.Conv2D(8, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        # Layer 3
        X = layers.Conv2D(16, self.filter_shape, activation="relu",
                    padding="same")(X)
        X = layers.Conv2D(16, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        # Layer 4
        X = layers.Conv2D(32, self.filter_shape, activation="relu",
                    padding="same")(X)
        X = layers.Conv2D(32, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        

        # Saving the shape of this KerasTensor and used it later
        last_conv_shape = X.shape

        # Adjust the shape
        X = layers.Flatten()(X)
        flatten_shape = X.shape

        # Fully Connected Layers
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dense(128, activation="relu")(X)
        X = layers.Dense(64, activation="relu")(X)
        X = layers.Dense(32, activation="relu")(X)

        encoder_output  = layers.Dense(self.bottelneck, activation="relu")(X)

        self.encoder = Model(
            encoder_input, encoder_output, name="encoder")

        if summary:
            self.encoder.summary()

        # --->>> Decoder
        decoder_input = layers.Input(
            shape=self.bottelneck, name="Latent Space")
        
        # Fully Connected Layers
        X = layers.Dense(32, activation="relu")(decoder_input)
        X = layers.Dense(64, activation="relu")(X)
        X = layers.Dense(128, activation="relu")(X)
        X = layers.Dense(512, activation="relu")(X)

        # adjust the shape
        X = layers.Dense(flatten_shape[1], activation="relu")(X)
        X = layers.Reshape(
            (last_conv_shape[1], last_conv_shape[2], last_conv_shape[3])
            )(X)

        # Convolutional Layers
        # Layer 1
        X = layers.UpSampling2D((2, 2))(X)  
        X = layers.Conv2D(32, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.Conv2D(32, self.filter_shape, activation="relu",
            padding="same")(X)
        
        # Layer 2    
        X = layers.UpSampling2D((2, 2))(X)  
        X = layers.Conv2D(16, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.Conv2D(16, self.filter_shape, activation="relu",
            padding="same")(X)       
        # Layer 3
        X = layers.UpSampling2D((2, 2))(X)  
        X = layers.Conv2D(8, self.filter_shape, activation="relu",
            padding="same")(X)
        X = layers.Conv2D(8, self.filter_shape, activation="relu",
            padding="same")(X)
        
        # Layer 4
        X = layers.UpSampling2D((2, 2))(X)  
        X = layers.Conv2D(1, self.filter_shape, activation="relu",
            padding="same")(X)
        
        #((top_crop, bottom_crop), (left_crop, right_crop))

        decoder_output = layers.Cropping2D(((0, X.shape[1] - self.grid_size[0]), (0, X.shape[2] - self.grid_size[1])))(X)

        self.decoder = Model(
            decoder_input, decoder_output, name="decoder")

        if summary:
            self.decoder.summary()

        #--->>> Autoencoder
        autoencoder_input = layers.Input(
            shape=self.grid_size, name="autoencoder_input")
        encoded_img = self.encoder(autoencoder_input)
        decoded_img = self.decoder(encoded_img)
        self.autoencoder = Model(
            autoencoder_input, decoded_img, name="autoencoder")

        if summary:
            self.autoencoder.summary()

    
    
    def train(self, train_data, test_data, epoch, EACH_N_EPOCH, batch_size, lr):

        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        self.autoencoder.compile(optimizer=opt, loss='mse')
        autoe_self = self
        class myCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch+1) % EACH_N_EPOCH == 0:
                    pred_test_data= autoe_self.autoencoder.predict(test_data)
                    numelems = 3  
                    numcols = 3
                    idx = np.round(np.linspace(0, len(test_data[:, 0, 0]) - 1, numelems)).astype(int)

                    for i in range(numelems):

                        ax = plt.subplot(numelems, numcols, numcols*i+1)                            
                        plt.imshow(
                            test_data[idx[i], :, :, 0],
                            cmap="jet",
                            interpolation="quadric",
                            vmin=0,
                            vmax=1)
                        plt.colorbar()
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)

                        if i == 0:
                            plt.title("original img")
                        ax = plt.subplot(numelems, numcols, numcols*i+2)
                        plt.imshow(
                            pred_test_data[idx[i], :, :, 0],
                            cmap="jet",
                            interpolation="quadric",
                            vmin=0,
                            vmax=1)
                        plt.jet()
                        plt.colorbar()
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        
                        if i == 0:
                            plt.title("Predicted")
                        ax = plt.subplot(numelems, numcols, numcols*i+3)
                        plt.imshow(
                            test_data[idx[i], :, :, 0] - pred_test_data[idx[i], :, :, 0],
                            cmap="jet",
                            interpolation="quadric",
                            vmin=0,
                            vmax=1)
                        plt.jet()
                        plt.colorbar()
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        if i == 0:
                            plt.title("Error")


                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()


        callback_list = [myCallback()]
          
        history = self.autoencoder.fit(
            x=train_data,
            y=train_data,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data),
            callbacks=callback_list
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        return history.history


