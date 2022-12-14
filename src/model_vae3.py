
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D , Flatten, Reshape, Conv2DTranspose, BatchNormalization, Conv1D, Input
from tensorflow.keras import Model
import tensorflow_probability as tfp


class Time_Freq_Autoencoder_Builder:
    
    def build(width, height, depth, filters=(16,32,64), latent_dim=128, kernel_size=5):

        prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(latent_dim), scale=1),
                        reinterpreted_batch_ndims=1)
        
        strides = 2
        
        input_shape = (height, width, depth)
        inputs = Input(shape = input_shape)
        
        chan_dim = -1
        
        x = Reshape(target_shape=(height,width))(inputs)
        
        for f in filters:
            
            x = Conv1D(f, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)
            
        x = Flatten()(x)
        
        x = Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim))(x)
        latents = tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))(x)

        
        encoder = Model(inputs, latents, name='encoder')
        
        latent_inputs = Input(shape=((latent_dim//2)*2))
        
        x = Dense((width*height), activation='relu')(latent_inputs)
        reshape_edge = int(np.sqrt((width*height)//filters[-1]))
        x = Reshape(target_shape=(reshape_edge, reshape_edge, filters[-1]))(x)
        
        for f in filters[::-1]:
            
            x = Conv2DTranspose(f, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
            x = BatchNormalization(axis=chan_dim)(x)
            
        x = Conv2DTranspose(depth, kernel_size=kernel_size, padding='same', activation='sigmoid')(x)

        # x = Flatten()(x)

        # x = tfp.layers.IndependentBernoulli(input_shape, tfp.distributions.Bernoulli.logits)(x)
        
        outputs = x
        
        decoder = Model(latent_inputs, outputs, name='decoder')
        
        autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
        
        return (encoder, decoder, autoencoder)
    

class Time_Freq_Autoencoder(tf.keras.Model):
    
    def __init__(self, image_width, image_height, image_depth=1, latent_dim=256, kernel_size=5, filters=(16,32,64)):
        super().__init__()
        
        self.encoder, self.decoder, self.autoencoder = Time_Freq_Autoencoder_Builder.build(width=image_width, height=image_height, depth=image_depth, latent_dim=latent_dim, kernel_size=kernel_size, filters=filters)
        
    def call(self, x):
        autoencoded = self.autoencoder(x)
        return autoencoded
