import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tqdm import tqdm
from gain_utils import rounding
from gain_utils import binary_sampler, uniform_sampler, sample_batch_index

class CosineAnnealingSchedule(LearningRateSchedule):
    def __init__(self, initial_lr, total_iterations, min_lr=0.0):
        """
        Cosine Annealing Learning Rate Scheduler.
        
        Args:
        - initial_lr: Initial learning rate
        - total_iterations: Total number of iterations
        - min_lr: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.total_iterations = total_iterations
        self.min_lr = min_lr

    def __call__(self, step):
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * step / self.total_iterations))
        return self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

class GAINImputer:
    def __init__(self, gain_parameters=None):
        """
        Initialize GAIN Imputer
        
        Args:
        - gain_parameters: Dictionary of parameters including:
          - batch_size: Batch size
          - hint_rate: Hint rate
          - alpha: Hyperparameter
          - iterations: Iterations
        """
        # Default parameters if not provided
        default_parameters = {
            'batch_size': 4,
            'hint_rate': 0.9,
            'alpha': 1,
            'iterations': 10000,
            'initial_lr': 0.0001,
            'min_lr': 1e-6
        }
        
        # Update default parameters with provided parameters
        self.gain_parameters = default_parameters
        if gain_parameters:
            self.gain_parameters.update(gain_parameters)
        
        # Placeholders for model and normalization parameters
        self.generator = None
        # self.max_value = 1.0
        self.data_mean = None
        self.data_std = None
    
    def fit(self, data_x):
        """
        Train the GAIN imputer on the input data
        
        Args:
        - data_x: original data with missing values
        
        Returns:
        - Imputed data
        """
        # Define mask matrix
        data_m = 1 - np.isnan(data_x)
        
        # Extract parameters
        batch_size = self.gain_parameters['batch_size']
        hint_rate = self.gain_parameters['hint_rate']
        alpha = self.gain_parameters['alpha']
        iterations = self.gain_parameters['iterations']
        initial_lr = self.gain_parameters['initial_lr']
        min_lr = self.gain_parameters['min_lr']
        
        # Other parameters
        no, dim = data_x.shape
        h_dim = int(dim)
        
        # Normalize input data
        data_x_normalized = np.nan_to_num(data_x, 0)
        # self.max_value = np.max(data_x_normalized)
        # if self.max_value != 0:
        #     data_x_normalized = data_x_normalized / self.max_value
        
        # Build generator with improved architecture
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(dim, activation='sigmoid', kernel_initializer='glorot_uniform')
        ])
        
        # Build discriminator with improved architecture
        discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(h_dim, kernel_initializer='glorot_uniform'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(dim, activation='sigmoid', kernel_initializer='glorot_uniform')
        ])

        # Cosine annealing for learning rate
        gen_lr_schedule = CosineAnnealingSchedule(initial_lr, iterations, min_lr)
        disc_lr_schedule = CosineAnnealingSchedule(initial_lr, iterations, min_lr)
        
        # Define optimizers
        gen_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr_schedule, beta_1=0.5, clipnorm=1.0)
        disc_optimizer = tf.keras.optimizers.Adam(learning_rate=disc_lr_schedule, beta_1=0.5, clipnorm=1.0)
        
        @tf.function
        def train_step(X_mb, M_mb, H_mb):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                X_mb = tf.cast(X_mb, tf.float32)
                M_mb = tf.cast(M_mb, tf.float32)
                H_mb = tf.cast(H_mb, tf.float32)
                
                # Generator forward pass
                G_sample = self.generator(tf.concat([X_mb, M_mb], axis=1))
                
                # Ensure G_sample is bounded
                G_sample = tf.clip_by_value(G_sample, 0, 1)
                
                Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
                
                # Discriminator forward pass
                D_prob = discriminator(tf.concat([Hat_X, H_mb], axis=1))
                
                # Add small epsilon to avoid log(0)
                eps = 1e-8
                
                # Modified loss functions
                D_loss_real = -tf.reduce_mean(M_mb * tf.math.log(D_prob + eps))
                D_loss_fake = -tf.reduce_mean((1 - M_mb) * tf.math.log(1. - D_prob + eps))
                D_loss = D_loss_real + D_loss_fake
                
                G_loss = -tf.reduce_mean((1 - M_mb) * tf.math.log(D_prob + eps))
                MSE_loss = tf.reduce_mean((M_mb * X_mb - M_mb * G_sample) ** 2) / (tf.reduce_mean(M_mb) + eps)
                
                G_loss += alpha * MSE_loss
            
            # Gradient computation and clipping
            gen_gradients = gen_tape.gradient(G_loss, self.generator.trainable_variables)
            disc_gradients = disc_tape.gradient(D_loss, discriminator.trainable_variables)
            
            gen_gradients = [tf.clip_by_norm(g, 1.0) for g in gen_gradients if g is not None]
            disc_gradients = [tf.clip_by_norm(g, 1.0) for g in disc_gradients if g is not None]
            
            gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
            
            return D_loss, G_loss, MSE_loss
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 2000
        patience_counter = 0
        
        # Training loop
        for it in tqdm(range(iterations)):
            batch_idx = sample_batch_index(no, batch_size)
            X_mb = data_x_normalized[batch_idx, :]
            M_mb = data_m[batch_idx, :]
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            
            D_loss, G_loss, MSE_loss = train_step(X_mb, M_mb, H_mb)

            # Early stopping check
            current_loss = D_loss.numpy() + G_loss.numpy()
            # if current_loss < best_loss:
            #     best_loss = current_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1
                
            # if patience_counter >= patience:
            #     print(f"Early stopping triggered at iteration {it + 1}")
            #     break
            
            if it % 1000 == 0:
                print(f"Iteration {it + 1}/{iterations} - D_loss: {D_loss.numpy():.4f}, G_loss: {G_loss.numpy():.4f}, MSE_loss: {MSE_loss.numpy():.4f}")
        
        return self
    
    def transform(self, data_x):
        """
        Transform input data using the trained generator
        
        Args:
        - data_x: input data with missing values
        
        Returns:
        - Imputed data
        """
        if self.generator is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create mask matrix
        data_m = 1 - np.isnan(data_x)
        
        # Normalize input data
        data_x_normalized = np.nan_to_num(data_x, 0)
        # if self.max_value != 0:
        #     data_x_normalized = data_x_normalized / self.max_value
        
        # Imputation
        Z_mb = uniform_sampler(0, 0.01, data_x_normalized.shape[0], data_x_normalized.shape[1])
        X_mb = data_x_normalized
        X_mb = data_m * X_mb + (1 - data_m) * Z_mb
        
        imputed_data = self.generator(tf.concat([X_mb, data_m], axis=1))
        imputed_data = data_m * data_x_normalized + (1 - data_m) * imputed_data.numpy()
        
        # Denormalize
        # if self.max_value != 0:
        #     imputed_data = imputed_data * self.max_value
        
        imputed_data = rounding(imputed_data, data_x)
        
        return imputed_data
    
    def fit_transform(self, data_x):
        """
        Fit the model and transform the data in one step
        
        Args:
        - data_x: input data with missing values
        
        Returns:
        - Imputed data
        """
        return self.fit(data_x).transform(data_x)