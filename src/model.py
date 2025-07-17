import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim, tire_compound_vocab_size, num_classes=20):
    # Input: [numerical_features..., tire_compound (int)]
    num_inputs = layers.Input(shape=(input_dim-1,), name='num_inputs')
    tire_input = layers.Input(shape=(1,), name='tire_compound')
    tire_emb = layers.Embedding(input_dim=tire_compound_vocab_size, output_dim=4, name='tire_emb')(tire_input)
    tire_emb_flat = layers.Flatten()(tire_emb)
    x = layers.Concatenate()([num_inputs, tire_emb_flat])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=[num_inputs, tire_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model 