import os
import cbox
import json
import time
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GlobalMaxPool2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications import mobilenet
from squeezenet import SqueezeNet

dataset_folder = 'dataset/demo'
training_file_json = dataset_folder + '/training_file.json'


def load_dataset(augmentation=False):
    X, y = [], []
    if not os.path.exists(dataset_folder) and not os.path.isdir(dataset_folder):
        raise ValueError("Dataset folder: {} does not exist.".format(dataset_folder))
    if augmentation:
        datagen = image.ImageDataGenerator(rotation_range=40, zca_whitening=True,
                                           zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        for cls_folder in os.listdir(dataset_folder):
            path = os.path.join(dataset_folder, cls_folder)
            if os.path.isdir(path):
                ori_file = np.array([
                    image.img_to_array(image.load_img(os.path.join(path, i))) for i in os.listdir(cls_folder)
                    if i != '.DS_Store' and 'augment' not in i])
                datagen.flow(ori_file, batch_size=len(ori_file), save_to_dir=path,
                             save_prefix='augment', save_format='jpg')

    training_list = [{'file': os.path.join(fo, f), 'class': os.path.basename(fo)}
                     for fo, void, flist in os.walk(dataset_folder)
                     for f in flist if fo != dataset_folder and f != '.DS_Store']
    with open(training_file_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(training_list, ensure_ascii=True))

    for t_file in training_list:
        img = image.load_img(t_file['file'], target_size=(224, 224))
        X.append(image.img_to_array(img))
        y.append(t_file['class'])
    X = np.array(X)  # .reshape(len(training_list), 128)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = onehot_encoder.fit_transform(integer_encoded)
    with open('dataset/label.pkl', 'wb') as f:
        f.write(pickle.dumps(label_encoder))
    # np.savetxt('dataset/feature.npy.gz', X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
    return x_train, x_test, y_train, y_test


def training_video_segmentation(x_train, x_test, y_train, y_test, use_model):
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    if use_model == 'squeezenet':
        batch_size = 210
        model = SqueezeNet(input_shape=(224, 224, 3), include_top=True, weights='imagenet')
        s_layer = Dense(y_test.shape[1], input_dim=1000, activation='softmax')(model.layers[-2].output)
    elif use_model == 'mobilenet':
        batch_size = 52
        model = mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
        s_layer = Dense(y_test.shape[1], input_dim=1024, activation='softmax')(model.output)

    for layer in model.layers:
        layer.trainable = True

    m = Model(model.input, s_layer)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(m.summary())
    m.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.3, shuffle=True)
    acc = m.evaluate(x_test, y_test, batch_size=batch_size)
    m.save('{}.model.h5'.format(use_model))
    print(acc)


def predicting_video_segmentation_mobilenet(img_path):
    pred_num = 1
    classes = ['choose', 'dead', 'end', 'not play', 'playing', 'start']
    if not os.path.exists(img_path):
        raise ValueError("File: {} does not exist.".format(img_path))
    if os.path.isdir(img_path):
        img_path = [os.path.join(img_path, i) for i in os.listdir(img_path) if i != '.DS_Store']
        pred_num = len(img_path)
    img = []
    for i in img_path:
        try:
            i_array = image.img_to_array(image.load_img(i, target_size=(224, 224)))
            img.append(i_array)
        except Exception:
            os.remove(i)
    img = np.array(img)
    pred = load_model('model.h5',
                      custom_objects={'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
    start = time.perf_counter()
    pd_rs = pred.predict(img)
    time_spend = time.perf_counter() - start
    rs = []
    for key, i in enumerate(img_path):
        rs.append({'key': int(i.split('.')[-2]),
                   'status': classes[int(np.argmax(pd_rs[key]))],
                   'detail': ", ".join(["%05f" % i for i in pd_rs[key]])})
    print()
    print('\t', '\t', '\t', 'choose', 'dead', 'end', 'not play', 'playing', 'start')
    for i in sorted(rs, key=lambda x: x['key']):
        print("%03d" % i['key'], '\t', "%08s" % i['status'], '\t', i['detail'])
    print(pred_num, "rows", "\t", time_spend, "sec.", "\t", pred_num // time_spend, "fps.")


@cbox.cmd
def main(operation, img_path='', use_model='squeezenet'):
    if operation == 'training-video-segmentation':
        x_train, x_test, y_train, y_test = load_dataset()
        training_video_segmentation(x_train, x_test, y_train, y_test, use_model=use_model)

    elif operation == 'predicting-video-segmentation':
        predicting_video_segmentation_mobilenet(img_path=img_path)


if __name__ == '__main__':
    cbox.main(main)
