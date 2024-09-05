import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
# train and create gan model start
def preprocess_gan_data(data):
    X = data.drop('key', axis=1).values  # Features
    y = data.drop('key',axis=1)['delay_time'].values  # Labels
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test,scaler

# Build the generator model
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(3, activation='linear'))  # Output: key_hold_time, key_flight_time, key_release_time
    return model

# Build the discriminator model
def build_discriminator(input_shape):
    model = Sequential()
    model.add(Dense(32, input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(16))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output: probability of being genuine
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build the GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
# Route for generating synthetic keystroke data
def generate_synthetic_gan_data(featured_path,model_gan_generator_filepath):
    latent_dim = 10
    generator = tf.keras.models.load_model(model_gan_generator_filepath)
    data = pd.read_csv(featured_path)
    X_train, X_test, y_train, y_test,scaler = preprocess_gan_data(data)
    print("X_train",X_train.tolist())
    
    latent_points = np.random.normal(0, 1, (len(X_train), latent_dim))
    
    generated_data = generator.predict(latent_points)
    
    # Build and train authentication model
    auth_model = Sequential([
        Dense(64, activation='relu', input_shape=(generated_data.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    auth_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    auth_model.fit(generated_data, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    predictions = auth_model.predict(generated_data)
    authenticate = np.mean(predictions)
    print("prediction", np.mean(predictions))
    
    print("authenticate",bool(authenticate))
    return predictions,authenticate

def create_and_train_gan_model(username,filepath):
    data = pd.read_csv(filepath)
    X_train, X_test, y_train, y_test,scaler = preprocess_gan_data(data)
    
    latent_dim = 10  # Dimensionality of the latent space
    generator = build_generator(latent_dim)
    discriminator = build_discriminator((X_train.shape[1],))
    gan = build_gan(generator, discriminator)
    
    # Training parameters
    epochs = 200
    batch_size = 32
    
    for epoch in range(epochs):
         # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_features = X_train[idx]
        fake_features = generator.predict(np.random.normal(0, 1, (batch_size, latent_dim)))
        d_loss_real = discriminator.train_on_batch(real_features, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_features, np.zeros((batch_size, 1)))
        
        # Train generator
        g_loss = gan.train_on_batch(np.random.normal(0, 1, (batch_size, latent_dim)), np.ones((batch_size, 1)))
        print(f"{epoch}/{epochs} [D loss: {0.5 * (d_loss_real[0] + d_loss_fake[0])} | D accuracy: {100 * 0.5 * (d_loss_real[1] + d_loss_fake[1])}%] [G loss: {g_loss}]")
       
    
    # Save the models
    model_dir = os.path.join("trained_models","gen_model_files")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_gan_generator_filepath = os.path.join(model_dir,f'{username}_keystroke_gan_generator.keras')
    model_gan_discriminator_filepath = os.path.join(model_dir,f'{username}_keystroke_gan_discriminator.keras')
    generator.save(model_gan_generator_filepath)
    discriminator.save(model_gan_discriminator_filepath)
    
    # generator.summary
    # discriminator.summary
    # generate_synthetic_data(filepath,model_gan_generator_filepath)
    return generator,discriminator,model_gan_generator_filepath,model_gan_discriminator_filepath
    
    
    
    
# train and create gan model end