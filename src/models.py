import tensorflow as tf
from tensorflow import keras

def early_stopping_cb(monitor='val_loss', patience=5, mode='min'):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                      patience=patience, 
                                                      mode=mode)
    return early_stopping

def multi_step_linklevel_model1(output_length, num_of_features, num_of_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=False))
    model.add(tf.keras.layers.Dense(output_length*num_of_features*num_of_classes,
                                    kernel_initializer=tf.initializers.zeros()))
    model.add(tf.keras.layers.Reshape([output_length, num_of_features, num_of_classes]))
    model.add(tf.keras.layers.Dense(num_of_classes, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def multi_step_linklevel_model2(output_length, num_of_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, activation='relu'))
    model.add(tf.keras.layers.RepeatVector(output_length))
    model.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_of_classes, activation='sigmoid')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()])
    return model

def evaluate_single_step():
    pass

def evaluate_multi_step():
    pass

def single_step_linklevel_model(num_of_classes, learning_rate):
    model = tf.keras.Sequential()
    model.add(keras.layers.LSTM(128, return_sequences=False))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()])
    return model
    # inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    # x = keras.layers.LSTM(128, return_sequences=False)(inputs)
    # x = keras.layers.Dropout(0.2)(x)
    # # outputs = keras.layers.Dense(1, name=f'Dense_1')(x)
    # outputs = keras.layers.Dense(num_of_classes, activation='sigmoid')(x)
    
def single_step_linklevel_multivar(n_timesteps, n_features, num_of_classes, learning_rate):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(keras.layers.RepeatVector(1))
    model.add(keras.layers.LSTM(200, activation='relu', return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(100, activation='relu')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(num_of_classes, activation='sigmoid')))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()])
    return model
 
# returns train, inference_encoder and inference_decoder models
def lstm_enc_dec_triplevel(n_total_features, latent_dim, past, future, n_deterministic_features, num_of_classes):
    # define training encoder
    encoder_inputs = tf.keras.Input(shape=(past, n_total_features))
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True, name='enc_train')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = keras.layers.Input(shape=(future, n_deterministic_features))
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, name='dec_train', return_state=True)
    x, _ , _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    # x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    decoder_dense = tf.keras.layers.Dense(num_of_classes, activation='sigmoid')
    decoder_outputs = decoder_dense(x)
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states, name='enc_inf_mod')

    # define inference decoder
    decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
    decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    x = tf.keras.layers.Dense(latent_dim, activation='relu', name='dec_inf_den1')(decoder_outputs)
    x = tf.keras.layers.Dense(latent_dim, activation='relu', name='dec_inf_den2')(x)
    feat_output = tf.keras.layers.Dense(n_deterministic_features, activation='softmax', name='dec_inf_den3')(x)
    decoder_outputs = decoder_dense(x)
    # TODO: Add another output for the classes? Since i dont pass the classes back
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs, feat_output] + decoder_states, name='dec_inf_mod')
    # return all models
    return model, encoder_model, decoder_model