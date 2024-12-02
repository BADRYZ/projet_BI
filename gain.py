import tensorflow as tf
import numpy as np
from tqdm import tqdm
from gain_utils import rounding
from gain_utils import binary_sampler, uniform_sampler, sample_batch_index


def gain(data_x, gain_parameters):
    '''Impute missing values in data_x
    
    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyperparameter
        - iterations: Iterations
        
    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape
    h_dim = int(dim)

    data_x = np.nan_to_num(data_x, 0)

    # Build generator with batch normalization
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(dim, activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    # Build discriminator with batch normalization
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(dim, activation='sigmoid', kernel_initializer='glorot_normal')
    ])

    # Define optimizers
    gen_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    disc_optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

    @tf.function
    def train_step(X_mb, M_mb, H_mb):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            X_mb = tf.cast(X_mb, tf.float32)
            M_mb = tf.cast(M_mb, tf.float32)
            H_mb = tf.cast(H_mb, tf.float32)

            # Generator forward pass
            G_sample = generator(tf.concat([X_mb, M_mb], axis=1))

            Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)

            # Discriminator forward pass
            D_prob = discriminator(tf.concat([Hat_X, H_mb], axis=1))

            # Losses
            D_loss = -tf.reduce_mean(
                M_mb * tf.math.log(D_prob + 1e-8) +
                (1 - M_mb) * tf.math.log(1. - D_prob + 1e-8)
            )
            G_loss = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + 1e-8))
            MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample) ** 2) / tf.reduce_mean(M_mb)

            G_loss += alpha * MSE_loss

        # Gradients and optimization
        gen_gradients = gen_tape.gradient(G_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(D_loss, discriminator.trainable_variables)

        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        return D_loss, G_loss, MSE_loss

    # Training loop
    for it in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        Z_mb = uniform_sampler(0, 0.001, batch_size, dim)
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        D_loss, G_loss, MSE_loss = train_step(X_mb, M_mb, H_mb)
        print(f"Iteration {it + 1}/{iterations} - D_loss: {D_loss.numpy():.4f}, G_loss: {G_loss.numpy():.4f}, MSE_loss: {MSE_loss.numpy():.4f}")

    # Imputation
    Z_mb = uniform_sampler(0, 0.001, no, dim)
    X_mb = data_x
    X_mb = data_m * X_mb + (1 - data_m) * Z_mb
    imputed_data = generator(tf.concat([X_mb, data_m], axis=1))
    imputed_data = data_m * data_x + (1 - data_m) * imputed_data.numpy()

    imputed_data = rounding(imputed_data, data_x)

    return imputed_data