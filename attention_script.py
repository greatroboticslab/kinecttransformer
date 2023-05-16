import os.path
import glob
import tensorflow as tf
import tensorflow.keras as keras
from kinect_learning import * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)

DATA_DIR = 'data'

def train_model(data_file_name, epochs):
    # 1. Load Data
    file_path = os.path.join(DATA_DIR, data_file_name)
    data_collection = joints_collection(data_file_name.rstrip('.csv'))
    noise = False
    data = load_data_multiple_dimension(
        file_path,
        data_collection,
        noise
    )
    (train_x, train_y), (val_x, val_y) = create_datasets(data['positions'], data['labels'])
    
    query_input = keras.Input(shape=train_x[0].shape, dtype='float64')
    value_input = keras.Input(shape=train_x[0].shape, dtype='float64')

    convolution_layer = keras.layers.DepthwiseConv1D(
        kernel_size=4,
        padding='same'
    )

    # 2. Build Model With Correct Shape
    convolved_query = convolution_layer(query_input)
    convolved_value = convolution_layer(value_input)
    
    attention_layer = keras.layers.Attention()(
        [convolved_query, convolved_value]
    )

    flatten_layer = keras.layers.Flatten()(attention_layer)

    dense_layer = keras.layers.Dense(32, kernel_initializer='he_normal')(flatten_layer)

    model = keras.Model(
        inputs=[query_input, value_input],
        outputs=[dense_layer],
        name='attention-model'
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalCrossentropy(), 'accuracy']
    )
    
    # 3. Train Model With Data
    history = model.fit(
        (train_x, train_x),
        train_y,
        validation_data=((val_x, val_x), val_y),
        epochs=epochs,
        callbacks=[]
    )
    model.predict((val_x, val_x)) # Make sure this works.
    return model, history

    
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
    FILE_NAMES = list(map(os.path.basename, glob.glob('./data/*.csv')))
    print('Training on files: {}'.format(FILE_NAMES))

    TRAINING_ATTEMPTS = 20
    EPOCHS=200
    RESULT = {
        file_name: [train_model(file_name, epochs=EPOCHS) for _ in range(TRAINING_ATTEMPTS)]
        for file_name in FILE_NAMES
    }
    BEST_RESULTS = {
        file_name: max(trained_models, key=lambda model: model[1].history['val_accuracy'])
        for file_name, trained_models in RESULT.items()
    }
    for DATA_FILE, BEST_RESULT in BEST_RESULTS.items():
        print(DATA_FILE, BEST_RESULT[1].history['val_accuracy'][-1])
