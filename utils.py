import os
import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import keras.backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


def data_augmentation(folder, gen_pics, results_folder=None):
    if not os.path.isdir(folder):
        assert ValueError, "%s is not a folder." % folder
    datagen = image.ImageDataGenerator(rotation_range=80, zoom_range=0.3, horizontal_flip=True, fill_mode='nearest')
    for f in os.listdir(folder):
        if f != '.DS_Store' and 'augment' not in f:
            filename = f[:-4]
            img_path = os.path.join(folder, f)
            try:
                img = np.array(image.img_to_array(image.load_img(img_path)))
            except TypeError:
                print("Remove ", img_path)
                os.remove(img_path)
            img = img.reshape(1, *img.shape)

            i = 0
            for _ in datagen.flow(img, save_to_dir=results_folder or folder,
                                  save_prefix='%s_augmentation' % filename, save_format='jpg'):
                i += 1
                if i > gen_pics:
                    break


def load_class_labels(results_folder, model_path):
    model_dir = os.path.join(os.path.dirname(model_path))
    label_path = os.path.join(model_dir, "labels.json") \
        if os.path.exists(os.path.join(model_dir, "labels.json")) \
        else os.path.join(results_folder, "labels.json")
    classes = json.load(open(label_path, "r", encoding="utf-8"))
    return classes


def load_dataset(dataset_folder, results_folder, target_size, img_preload=False):
    X, ori_y = [], []
    label_file = os.path.join(results_folder, "labels.json")
    if not os.path.exists(dataset_folder) and not os.path.isdir(dataset_folder):
        raise ValueError("Dataset folder: {} does not exist.".format(dataset_folder))

    training_list = [{'file': os.path.join(fo, f), 'class': os.path.basename(fo)}
                     for fo, void, flist in os.walk(dataset_folder)
                     for f in flist if fo != dataset_folder and f != '.DS_Store']

    for t_file in training_list:
        if img_preload:
            img = image.load_img(t_file['file'], target_size=target_size)
            X.append(image.img_to_array(img))
        else:
            X.append(t_file['file'])
        ori_y.append(t_file['class'].split(','))
    X = np.array(X)  # .reshape(len(training_list), 128)

    # integer encode
    binarizer = MultiLabelBinarizer()
    y = binarizer.fit_transform(ori_y)
    classes = list(binarizer.classes_)

    # Remove column
    removable_cat = set([i for i in ori_y if '[0]' in i])     # if folder has "[0]" str, the value will be all in zero.
    removable_col = [classes.index(i) for i in removable_cat]
    for c in sorted(removable_col, reverse=True):
        y = np.delete(y, c, 1)
        classes.pop(c)

    # Split array
    y_softmax_cat = set([k for i in ori_y for k in i if '-softmax-' in k])
    y_softmax_col = sorted([classes.index(i) for i in y_softmax_cat])
    y_softmax = y[::, min(y_softmax_col):max(y_softmax_col)+1]
    y_sigmoid_cat = set([k for i in ori_y for k in i if '-sigmoid-' in k])
    y_sigmoid_col = sorted([classes.index(i) for i in y_sigmoid_cat])
    y_sigmoid = y[::, min(y_sigmoid_col):max(y_sigmoid_col)+1]
    y = np.concatenate((y_softmax, y_sigmoid), axis=1)
    classes = [i[9:] for i in classes]
    classes = [classes[min(y_softmax_col):max(y_softmax_col)], classes[min(y_sigmoid_col):max(y_sigmoid_col)]]
    with open(label_file, 'w') as f:
        f.write(json.dumps(classes))
    return X, y, training_list


def generate_batch(img_path, y, batch_size):
    batch_features = np.zeros((batch_size, 64, 64, 3))
    batch_labels = np.zeros((batch_size, 1))

    while True:
        for i in range(batch_size):
            # choose random index in features
            index = random.choice(len(features), 1)
            batch_features[i] = some_processing(features[index])
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


class DataGenerator:
    """Generates data for Keras"""
    def __init__(self, folder, dim_x=32, dim_y=32, dim_z=32, batch_size=32, shuffle=True):
        """Initialization"""
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.folder = folder
        self.batch_size = batch_size
        self.shuffle = shuffle

    def generate(self, labels, list_IDs):
        """Generates batches of samples"""
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                X, y = self.__data_generation(labels, list_IDs_temp)

                yield X, y

    def __get_exploration_order(self, list_IDs):
        """Generates order of exploration"""
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, labels, list_IDs_temp):
        """Generates data of batch_size samples"""  # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z, 1))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            X[i, :, :, :, 0] = np.load(ID + '.npy')

            # Store class
            y[i] = labels[ID]

        return X, sparsify(y)


def training_callback(results_folder, use_model="squeezenet"):
    tensorboard_folder = os.path.join(results_folder, "tensorboard_logs")
    if not os.path.isdir(tensorboard_folder):
        os.mkdir(tensorboard_folder)
    model_checkpoing = ModelCheckpoint(
        os.path.join(results_folder, "%s.model.{epoch:03d}epoch.{val_fc/softmax_acc:.3f}acc.h5" % use_model),
        monitor='val_fc/softmax_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.01)
    early_stop = EarlyStopping(monitor='val_fc/softmax_loss', min_delta=0.0001, patience=20, mode='auto', verbose=1)
    tensorboard = TensorBoard(log_dir=tensorboard_folder, write_graph=True, write_images=True)
    # , embeddings_freq=1, embeddings_layer_names='fc/softmax')
    return [reduce_lr, model_checkpoing, early_stop, tensorboard]


def export_to_tf_model(model_path, export_folder):
    # reset session
    K.clear_session()
    sess = tf.Session()
    K.set_session(sess)

    # disable loading of learning nodes
    K.set_learning_phase(0)

    # load model
    model = load_model(model_path)
    config = model.get_config()
    weights = model.get_weights()

    # export saved model
    builder = saved_model_builder.SavedModelBuilder(export_folder)

    signature = predict_signature_def(inputs={'input': model.input},
                                      outputs={'output': model.output})

    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                             signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
        builder.save()


def export_to_coreml_model(model_path, export_folder):
    pass
