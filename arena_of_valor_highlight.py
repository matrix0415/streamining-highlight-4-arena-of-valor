import os
import cbox
import time
import arrow
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K
from keras.preprocessing import image
from keras.applications import mobilenet, inception_v3
from keras.models import Model, load_model
from keras.layers import Dense, Dot, GlobalMaxPool2D, Concatenate, GaussianDropout
from moviepy.editor import VideoFileClip, ImageSequenceClip
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from squeezenet import SqueezeNet
from utils import load_dataset, load_class_labels, training_callback, data_augmentation


target_size = (224, 224)
target_epoches = 200
gpu_avaliable = True if K.tensorflow_backend._get_available_gpus() else False


def training_video_segmentation(x_train, x_test, y_train, y_test, use_model, results_folder):
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    batch_size = 0

    if use_model == 'squeezenet':
        batch_size = 220
        m = SqueezeNet(input_shape=(*target_size, 3), weights='imagenet', classes=y_test.shape[1])
    elif use_model == 'mobilenet':
        batch_size = 52
        model = mobilenet.MobileNet(input_shape=(*target_size, 3), include_top=False, weights='imagenet', pooling='avg')
        s_layer = Dense(y_test.shape[1], input_dim=1024, activation='softmax', name='fc/softmax')(model.output)
        m = Model(model.input, s_layer)
    elif use_model == 'inceptionv3':
        batch_size = 38
        model = inception_v3.InceptionV3(input_shape=(*target_size, 3), include_top=False, weights='imagenet', pooling='avg')
        s_layer = Dense(y_test.shape[1], input_dim=1024, activation='softmax', name='fc/softmax')(model.output)
        m = Model(model.input, s_layer)
    else:
        assert ValueError("Model: {} currently not support.".format(use_model))

    for layer in m.layers:
        layer.trainable = True

    if not gpu_avaliable:
        batch_size = 256

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(m.summary())
    start = time.perf_counter()

    m.fit(x_train, y_train,
          epochs=target_epoches,
          batch_size=batch_size,
          validation_split=0.2,
          shuffle=True,
          callbacks=training_callback(results_folder=results_folder, use_model=use_model))

    time_spend = time.perf_counter() - start
    m.save('{}/{}.model.h5'.format(results_folder, use_model))
    print("Training Spend: ", time_spend/60)
    print("Save in folder: ", results_folder)
    model_evaluate(model_path=results_folder, x_test=x_test, y_test=y_test)


def predicting_video_segmentation(model_path, img_path, results_folder):
    custom_obj = None
    classes = load_class_labels(model_path=model_path, results_folder=results_folder)
    if not os.path.exists(img_path):
        raise ValueError("File: {} does not exist.".format(img_path))

    # Load model
    if 'mobilenet' in model_path:
        custom_obj = {'relu6': mobilenet.relu6, 'DepthwiseConv2D': mobilenet.DepthwiseConv2D}
    pred = load_model(model_path, custom_objects=custom_obj)

    # Load image files
    if os.path.isdir(img_path):
        img_path = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        img_path = [i for i in img_path if os.path.isfile(i)]

    rs = []
    pd_rs = []
    pred_num = len(img_path)

    if 3000 > pred_num:
        img = []
        for i in img_path:
            try:
                i_array = image.img_to_array(image.load_img(i, target_size=target_size))
                img.append(i_array)
            except Exception:
                print("Remove: ", i)
                os.remove(i)
        img = np.array(img)

        start = time.perf_counter()

        # Predict
        pd_rs = pred.predict(img)
        time_spend = time.perf_counter() - start

        for key, i in enumerate(img_path):
            rs.append({'key': int(i.split('.')[-2]),
                       'path': os.path.basename(i),
                       'status': classes[int(np.argmax(pd_rs[key]))],
                       'detail': ", ".join(["%05f" % i for i in pd_rs[key]])})
        print()
        print("Prediction Result Shape:", pd_rs.shape)
    else:
        start = time.perf_counter()
        for i in img_path:
            try:
                i_array = image.img_to_array(image.load_img(i, target_size=target_size))
                i_array = i_array.reshape(1, *i_array.shape)
                # Predict
                pd_rs.append(pred.predict(i_array)[0])

            except Exception:
                print("Remove: ", i)
                os.remove(i)
        time_spend = time.perf_counter() - start
        pd_rs = np.array(pd_rs)

        for key, i in enumerate(img_path):
            rs.append({'key': int(i.split('.')[-2]),
                       'path': os.path.basename(i),
                       'status': classes[int(np.argmax(pd_rs[key]))],
                       'detail': ", ".join(["%05f" % i for i in pd_rs[key]])})
        print()
        print("Prediction Result Shape:", pd_rs.shape)

    print('\t', '\t', '\t', classes)
    for i in sorted(rs, key=lambda x: x['key']):
        print("%03d" % i['key'], '\t', i['path'][-40:], '\t', "%08s" % i['status'], '\t', i['detail'])
    print(pred_num, "rows", "\t", time_spend, "sec.", "\t", pred_num // time_spend, "fps.", "\t", arrow.now())
    print()
    return img_path, pd_rs


def model_evaluate(model_path, x_test, y_test):
    if os.path.isfile(model_path):
        models = [model_path]
    else:
        models = [os.path.join(model_path, i) for i in os.listdir(model_path) if '.h5' in i]
    for model in models:
        print(os.path.basename(model))

        m = load_model(model)
        acc = m.evaluate(x_test, y_test)
        y_pred = m.predict(x_test)

        y_test_argmax = [np.argmax(i) for i in y_test]
        y_pred_argmax = [np.argmax(i) for i in y_pred]

        print("Keras Accuracy: ", acc)
        print("Recall", recall_score(y_test_argmax, y_pred_argmax, average="weighted"))
        print("Precision", precision_score(y_test_argmax, y_pred_argmax, average="weighted"))
        print("f1-score", f1_score(y_test_argmax, y_pred_argmax, average="weighted"))
        print()


def training_characters_identify(x_train, y_train, approx_x_test, approx_y_test, status_x_test, status_y_test):
    y_train = []
    model1 = SqueezeNet(input_shape=(50, 50, 3), include_top=False, weights='imagenet')
    model2 = SqueezeNet(input_shape=(275, 75, 3), include_top=False, weights='imagenet')
    dropout = GaussianDropout(0.2)(model2.output)
    gmp1 = GlobalMaxPool2D()(model1.output)
    gmp2 = GlobalMaxPool2D()(dropout)
    approx = Dot(1, normalize=True)([gmp1, gmp2])
    status = Dense(2, activation="sigmoid")(gmp2)
    concate = Concatenate()([approx, status])
    model = Model([model1.input, model2.input], concate)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    start = time.perf_counter()

    model.fit(x_train, y_train,
              epochs=target_epoches, batch_size=128, validation_split=0.3, shuffle=True,
              callbacks=training_callback(results_folder=results_folder))

    time_spend = time.perf_counter() - start
    acc = model.evaluate(x_test, y_test)
    y_pred = model.predict(x_test)
    print(acc, y_test.shape, y_pred.shape)


def predicting_characters_identify():
    pass


@cbox.cmd
def main(operation='', path='', model_path='', use_model='squeezenet', classname="", pickup_mode="copy"):
    results_folder = os.path.join("results", arrow.now().format("YYYY-MM-DD-HH-mm-ss") + "-" + operation)
    os.makedirs(results_folder, exist_ok=True)
    [os.rmdir(os.path.join("results", i)) for i in os.listdir("results")
     if os.path.join("results", i) != results_folder and
     len(os.listdir(os.path.join("results", i))) == 0]

    if operation == '':
        print("Operation List: ")
        print("image-augmentation: --path")
        print("training-video-segmentation: --path, --use-model")
        print("predicting-video-segmentation: --model-path, --path")
        print("pickup-prediction-result: --model-path, --path, --classname, --pickup-mode")
        print("video-clip: --path")
        print("generate-prediction-result-to-video: --model-path, --path")

    if operation == 'image-augmentation':
        data_augmentation(folder=path, gen_pics=1, results_folder=results_folder)

    if operation == 'training-video-segmentation':
        if not os.path.exists(path):
            assert ValueError, "Dataset folder is not exist: " + path
        X, y, _ = load_dataset(dataset_folder=path, results_folder=results_folder, img_preload=True, target_size=target_size)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        training_video_segmentation(x_train, x_test, y_train, y_test, use_model=use_model, results_folder=results_folder)

    if operation == 'predicting-video-segmentation':
        predicting_video_segmentation(model_path=model_path, img_path=path, results_folder=results_folder)
        os.rmdir(results_folder)

    if operation == 'pickup-prediction-result':
        classes = load_class_labels(model_path=model_path, results_folder=results_folder)
        if classname not in classes:
            assert ValueError, "Class name isn't exist! Available: " + ", ".join(classes)
        cls_index = classes.index(classname)
        img_path, pred = predicting_video_segmentation(img_path=path, model_path=model_path, results_folder=results_folder)
        img_path = [img_path[key] for key, i in enumerate(pred) if np.argmax(i) == cls_index]
        if img_path:
            for path in img_path:
                filebasename = os.path.join(results_folder, os.path.basename(path))[:-4]+".jpg"       # remove "jpg"
                if pickup_mode == "copy":
                    shutil.copy(path, filebasename)
                elif pickup_mode == "split":
                    img = Image.open(path)
                    img.crop((375, 105, 670, 180)).save(filebasename+".kill.jpg")       # 485, 105, 560, 180
                    img.crop((610, 105, 905, 180)).save(filebasename+".killed.jpg")     # 720, 105, 795, 180

    if operation == 'generate-prediction-result-to-video':
        if not model_path or not path:
            assert ValueError, "Require --model-path & --path"
        if os.path.isfile(path):
            basename = os.path.basename(path)[:-4]
            clip = VideoFileClip(path)
            clip.write_images_sequence(nameformat="{}/{}.frame.%05d.jpg".format(results_folder, basename), fps=10)
            path = results_folder

        print()
        print("Len: {}, Predicting...".format(len(os.listdir(path))))
        print()
        classes = load_class_labels(model_path=model_path, results_folder=results_folder)
        color_index = np.random.choice(range(150, 255, 5), (len(classes), 3), replace=False)
        font = ImageFont.truetype("OpenSans-Semibold.ttf", 40)

        img_path, pred = predicting_video_segmentation(img_path=path, model_path=model_path,
                                                       results_folder=results_folder)

        for key, p in enumerate(sorted(img_path)):
            cls_index = int(np.argmax(pred[key]))
            s = classes[cls_index] + " : " + str(np.max(pred[key]))
            img = Image.open(p)
            draw = ImageDraw.Draw(img)
            draw.text((100, 50), text=s, fill=tuple(color_index[cls_index]), font=font)
            img.resize((int(img.size[0]/2), int(img.size[1]/2)), Image.ANTIALIAS)
            img.save(p)
            if key % 1000 == 0:
                print("Proccessing: {}/{}.".format(key, len(img_path)))

        fname = os.path.basename(results_folder)
        clip = ImageSequenceClip(path, fps=25)
        clip.write_videofile(os.path.join(results_folder, fname + ".mp4"))
        [os.remove(i) for i in img_path]

    if operation == 'video-clip':
        if not os.path.isfile(path):
            assert ValueError, "Video File doesn't exist. " + path
        basename = os.path.basename(path)[:-4]
        clip = VideoFileClip(path)
        clip.write_images_sequence(nameformat="{}/{}.frame.%05d.jpg".format(results_folder, basename), fps=1)


if __name__ == '__main__':
    cbox.main(main)
