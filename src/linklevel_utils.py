import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Dropout, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from tqdm import tqdm

def setup_simple_lstm_generator(num_features, num_classes, learning_rate=1e-4):
    # define model
    model = tf.keras.Sequential()
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["sparse_categorical_accuracy"],
    )

    input_shape = (None, None, num_features)
    model.build(input_shape)
    return model

# def setup_lstm_encoder_decoder_from_weights_file(mpath, n_features, num_of_classes, n_lstm_units=256):
#     num_inputs = Input(shape=(None, n_features), name='numerical_input')

#     encoder_lstm = LSTM(n_lstm_units, return_sequences=False, return_state=True, name='encoder_lstm')
#     _, state_h, state_c = encoder_lstm(num_inputs)
#     encoder_states = [state_h, state_c]

#     decoder_inputs = Input(shape=(None, n_features), name='decoder_input')
#     decoder_lstm = LSTM(n_lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
#     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
#     dense_layer_1 = Dense(64, activation='relu')
#     dense_layer_2 = Dense(32, activation='relu')
#     # dense_layer_3 = Dense(32, activation='relu')
#     decoder_outputs = dense_layer_1(decoder_outputs)
#     decoder_outputs = dense_layer_2(decoder_outputs)
#     # decoder_outputs = dense_layer_3(decoder_outputs)

#     # Output 1: For forecasting class
#     decoder_dense_clas = Dense(num_of_classes, activation='softmax', name='decoder_dense_clas')
#     dec_out_clas = decoder_dense_clas(decoder_outputs)

#     model = Model(inputs=[num_inputs, decoder_inputs], outputs=[dec_out_clas])

#     old_model = model.load_weights(mpath)

#     num_inputs = old_model.get_layer('numerical_input').input
#     decoder_inputs = old_model.get_layer('decoder_input').input
#     _, state_h, state_c = old_model.get_layer('encoder_lstm').output
#     encoder_states = [state_h, state_c]
    
#         # Inference (sampling)
#     encoder_predict_model = Model(num_inputs, encoder_states)

#     decoder_state_input_h = Input(shape=(n_lstm_units,))
#     decoder_state_input_c = Input(shape=(n_lstm_units,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]

#     decoder_outputs = dense_layer_1(decoder_outputs)
#     decoder_outputs = dense_layer_2(decoder_outputs)
#     dec_out_clas = decoder_dense_clas(decoder_outputs)

#     decoder_predict_model = Model([decoder_inputs] + decoder_states_inputs, [dec_out_clas] + decoder_states)
#     return encoder_predict_model, decoder_predict_model

def setup_lstm_encoder_decoder_from_model_file(mpath, num_of_classes, n_lstm_units=256):
    old_model = keras.models.load_model(mpath, compile=False)

    num_inputs = old_model.get_layer('numerical_input').input
    decoder_inputs = old_model.get_layer('decoder_input').input
    _, state_h, state_c = old_model.get_layer('encoder_lstm').output
    encoder_states = [state_h, state_c]
    
    # Inference (sampling)
    encoder_predict_model = Model(num_inputs, encoder_states)
    
    decoder_lstm = LSTM(n_lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    dense_layer_1 = Dense(64, activation='relu')
    dense_layer_2 = Dense(32, activation='relu')
    decoder_dense_clas = Dense(num_of_classes, activation='softmax', name='decoder_dense_clas')

    decoder_state_input_h = Input(shape=(n_lstm_units,))
    decoder_state_input_c = Input(shape=(n_lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    decoder_outputs = dense_layer_1(decoder_outputs)
    decoder_outputs = dense_layer_2(decoder_outputs)
    dec_out_clas = decoder_dense_clas(decoder_outputs)

    decoder_predict_model = Model([decoder_inputs] + decoder_states_inputs, [dec_out_clas] + decoder_states)
    return encoder_predict_model, decoder_predict_model

def setup_transformer_model():
    return None

@tf.autograph.experimental.do_not_convert
def timeseries_dataset_from_dataset_ED(df, label_slice, input_sequence_length, output_sequence_length, batch_size, 
                                    offset=0, feature_slice=slice(None, None, None), shuffle=True):
    if offset > 0:
        output_sequence_length = output_sequence_length + offset
        
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    
    # must be at least output_sequence_length
    ds = dataset.window(input_sequence_length + output_sequence_length, shift=1, drop_remainder=True)
    # This controls the number of rows to be considered per batch
    ds = ds.flat_map(lambda x: x.batch(input_sequence_length + output_sequence_length))
    if shuffle:
        shuffle_buffer_size = len(df)
        data = data.shuffle(shuffle_buffer_size, seed=42)
    
    def split_feature_label(x):
        labels = {}
        image = {}
        teacher_null = np.full((output_sequence_length, len(df.columns)), -1)
        teacher_forcing = x[slice(input_sequence_length + offset, input_sequence_length + output_sequence_length + offset, None)]
        image['numerical_input']      = x[:input_sequence_length:, feature_slice]
        # image['decoder_input']        = teacher_forcing
        image['decoder_input']        = teacher_null
        labels['decoder_dense_clas'] = x[slice(input_sequence_length + offset, input_sequence_length + output_sequence_length + offset, None), label_slice]
        return image, labels
    
    ds = ds.map(split_feature_label)
    return ds.batch(batch_size)

# Can add shuffle in the future
@tf.autograph.experimental.do_not_convert
def timeseries_dataset_from_dataset(df, feature_slice, label_slice, input_sequence_length, output_sequence_length, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    ds = dataset.window(input_sequence_length + output_sequence_length, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x).batch(input_sequence_length + output_sequence_length)
     
    def split_feature_label(x):
        return x[:input_sequence_length:, feature_slice], x[input_sequence_length:,label_slice]
     
    ds = ds.map(split_feature_label)
     
    return ds.batch(batch_size)

# Transformer model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    num_classes,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

def prepare_linklevel(df, train_dates=None, val_dates=None, test_dates=None,
                      cat_columns=None, num_columns=None, ohe_columns=None,
                      feature_label='load', time_feature_used='arrival_time', scaler='minmax'):
    categorical_features = cat_columns
    numerical_features = num_columns
    all_used_columns = categorical_features + numerical_features

    ohe_encoder = OneHotEncoder()
    ohe_encoder = ohe_encoder.fit(df[ohe_columns])
    df[ohe_encoder.get_feature_names_out()] = ohe_encoder.transform(df[ohe_columns]).toarray()
    df = df.drop(ohe_columns, axis=1)
        
    train_df = df[(df[time_feature_used] >= train_dates[0]) &\
                  (df[time_feature_used] <= train_dates[1])]

    val_df = df[(df[time_feature_used] >= val_dates[0]) &\
                (df[time_feature_used] <= val_dates[1])]

    test_df = df[(df[time_feature_used] >= test_dates[0]) &\
                 (df[time_feature_used] <= test_dates[1])]
    
    print("Train df: ", train_df.shape)
    print("Val df: ", val_df.shape)
    print("Test df: ", test_df.shape)

    drop_cols = list(filter(lambda x: x not in all_used_columns, df.columns.tolist()))
    print(f"Columns to drop: {drop_cols}")
    # train_df = train_df.drop(drop_cols, axis=1)
    # val_df = val_df.drop(drop_cols, axis=1)
    # test_df = test_df.drop(drop_cols, axis=1)

    label_encoders = {}
    for col in [col for col in categorical_features if col != feature_label]:
        encoder = LabelEncoder()
        encoder = encoder.fit(df[col].unique())
        label_encoders[col] = encoder
        train_df[col] = encoder.transform(train_df[col])
        val_df[col] = encoder.transform(val_df[col])
        test_df[col] = encoder.transform(test_df[col])

    # Scaling numerical variables
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler = scaler.fit(train_df[num_columns])
    train_df[num_columns] = scaler.transform(train_df[num_columns])
    val_df[num_columns] = scaler.transform(val_df[num_columns])
    test_df[num_columns] = scaler.transform(test_df[num_columns])

    return ohe_encoder, label_encoders, scaler, train_df, val_df, test_df

'''
# usage:
ds = tf.data.Dataset.from_tensor_slices(tensor)
ds = timeseries_dataset_from_dataset(ds, slice(7, None, None), input_sequence_length=4, output_sequence_length=2, batch_size=256)
'''
def timeseries_dataset_from_dataset(dataset, label_slice, input_sequence_length, output_sequence_length, batch_size):
    print(tf.__version__)
    """
    A function to convert tensors into properly batched datasets for RNNs.
    From: https://mobiarch.wordpress.com/2020/11/13/preparing-time-series-data-for-rnn-in-tensorflow/

    Paramaters
    ----------
    dataset : dataset
        An array like Tensor. Different datasets in Tensorflow deliver wildly different types of data structure. 
        For example, tf.data.experimental.CsvDwataset delivers each row as a tuple of scalar tensors. 
        This must be mapped to an array like Tensor. For DataFrame: tensor = tf.convert_to_tensor(pd.DataFrame)
    label_slice : slice object
        A slice object to tell the function how to separate the labels from the whole feature set. 
        For example, A tensor with 3 feature columns can use a slice object slice(3, None, None) 
        to fish out the labels.
    input_sequence_length : int
        Number of past data/rows to use as input.
    output_sequence_length : int
        Number of future data/rows to predict. Cannot be greater than the `input_sequence_length`
    batch_size: int
        Number of batches in the dataset. Will be used by the `.fit()` as the batch_size.

    Return
    -------
    dataset : dataset
        Each item in the dataset will have the following dimensions for RNN use: 
        training = (batch_size, input_sequence_length, number of features)
        target = (batch_size, output_sequence_length, number of targets)
    """
    # ds = tf.data.Dataset.from_tensor_slices(dataset)
    ds = dataset.window(input_sequence_length + output_sequence_length, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda x: x).batch(input_sequence_length + output_sequence_length)
     
    def split_feature_label(x):
        return x[:input_sequence_length], x[input_sequence_length:,label_slice]
     
    ds = ds.map(split_feature_label)
     
    return ds.batch(batch_size)

# Input: Entire APC dataset
# For now only uses single column feature
# train_test_split: Datetime ranges
# time_feature_used: arrival_time

@tf.autograph.experimental.do_not_convert
def create_dataset(df, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size, shuffle=False, with_label=True):
    # Feel free to play with shuffle buffer size
    shuffle_buffer_size = len(df)
    # shuffle_buffer_size = 1024
    # Total size of window is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size

    data = tf.data.Dataset.from_tensor_slices(df.values)

    # Selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))

    # Shuffling data (seed=Answer to the Ultimate Question of Life, the Universe, and Everything)
    # https://stackoverflow.com/questions/57041305/keras-shuffling-dataset-while-using-lstm
    if shuffle:
        data = data.shuffle(shuffle_buffer_size, seed=42)

    # Extracting past features + deterministic future + labels
    if with_label:
        data = data.map(lambda k: ((k[:-forecast_size],
                                    k[-forecast_size:, -n_deterministic_features:]),
                                    k[-forecast_size:, 0]))
    else:
        data = data.map(lambda k: (k[:-forecast_size],
                                   k[-forecast_size:, -n_deterministic_features:]))

    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def simple_batching(test_df, WINDOW=5):
    # Prepare dataset (use past 5 to predict next 1 each time), sliding windows
    # Get a window 5 data points/stops as basis for the next stop.
    test_df = test_df.reset_index(drop=True)
    starts = np.array(test_df.index[0::WINDOW + 1])
    ends = np.array(test_df.index[WINDOW::WINDOW + 1])

    # Since using iloc or ranges is open on right
    to_predict = ends

    # Zip starts and ends to create a list of tuples
    windows = list(zip(starts, ends))

    x_stack = []
    y_stack = []
    for i in tqdm(range(len(to_predict))):
    # for i in tqdm(range(10)):
        start = windows[i][0]
        end   = windows[i][1]
        x = test_df[start:end].to_numpy()
        y = test_df.iloc[to_predict[i]].y_class
        x_stack.append(x)
        y_stack.append(y)

    X = np.vstack(x_stack)
    X = X.reshape(len(to_predict), x.shape[0], x.shape[1])
    return X, np.array(y_stack)