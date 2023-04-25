import argparse
import tensorflow as tf
import tensorflow.keras as keras
from os.path import join
from kinect_learning import * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)


QUERY_INPUT = keras.Input(shape=(14, 3), dtype='float64')
VALUE_INPUT = keras.Input(shape=(14, 3), dtype='float64')

CONVOLUTION_LAYER = keras.layers.DepthwiseConv1D(
    kernel_size=4,
    padding='same'
)

CONVOLVED_QUERY = CONVOLUTION_LAYER(QUERY_INPUT)
CONVOLVED_VALUE = CONVOLUTION_LAYER(VALUE_INPUT)

ATTENTION_LAYER = keras.layers.Attention()(
    [CONVOLVED_QUERY, CONVOLVED_VALUE]
)

FLATTEN_LAYER = keras.layers.Flatten()(ATTENTION_LAYER)

DENSE_LAYER = keras.layers.Dense(32)(FLATTEN_LAYER)

MODEL = keras.Model(
    inputs=[QUERY_INPUT, VALUE_INPUT],
    outputs=[DENSE_LAYER],
    name='attention-model'
)

MODEL.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd'
)


    
def create_datasets(x, y, test_size=0.4):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    print(x.shape)
    print(y.shape)
    # Shuffle
    indices = np.array(range(0, x.shape[0]))
    #np.random.shuffle(indices)
    x = np.take(x, indices, axis=0)
    y = np.take(y, indices)

    division = x.shape[0] % 200
    actual_length =  x.shape[0] - division
    x = x[0:actual_length,:]
    y = y[0:actual_length]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)


    return (x_train, y_train), (x_valid, y_valid)
    
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--data-file-name', type=str)
    ARGS = PARSER.parse_args()
    DATA_DIR = 'data'
    FILE_NAME = ARGS.data_file_name #'bending.csv'
    print(f'Running on {FILE_NAME}')
    FILE_PATH = join(DATA_DIR, FILE_NAME)
    
    print('Data loading start...')
    print(FILE_NAME.rstrip('.csv'))
    COLLECTION = joints_collection(FILE_NAME.rstrip('.csv'))
    assert COLLECTION
    print("Printing scores of small collection...")
    print("Collection includes", COLLECTION)
    print("Printing scores of small collection with noise data...")
    NOISE = False
    DATA = load_data_multiple_dimension(
        FILE_PATH,
        COLLECTION,
        NOISE
    )
    (TRAIN_X, TRAIN_Y), (VAL_X, VAL_Y) = create_datasets(DATA['positions'], DATA['labels'])
    MODEL.fit(
        tf.data.Dataset.zip(tf.data.Dataset.zip((tf.data.Dataset(TRAIN_X), tf.data.Dataset(TRAIN_X))), tf.data.Dataset(TRAIN_Y)).batch(32),

        validation_data=((VAL_X, VAL_X), VAL_Y),
        callbacks=[]#EARLY_STOPPING_CB, TENSOR_BOARD_CB]
    )
    MODEL.predict(VAL_X)
