import os
import cbox
import time
import arrow
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Dense
from keras.models import Model, load_model
from moviepy.editor import VideoFileClip, ImageSequenceClip
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from squeezenet import SqueezeNet
from utils import load_dataset, load_class_labels, training_callback, data_augmentation


target_size = (224, 224)
target_epoches = 200
gpu_avaliable = True if K.tensorflow_backend._get_available_gpus() else False


def training_video_segmentation(x_train, x_test, y_train, y_test, results_folder):
    print(x_train.shape, y_train.shape if type(y_train) == np.ndarray else len(y_train))
    print(x_test.shape, y_test.shape if type(y_train) == np.ndarray else len(y_train))
    y_train = [y_train[::, :6], y_train[::, 6:]]
    y_test = [y_test[::, :6], y_test[::, 6:]]

    batch_size = 220
    m = SqueezeNet(input_shape=(*target_size, 3), weights='imagenet', classes=11)
    sof = Dense(6, activation='softmax', name='fc/softmax')(m.layers[-2].output)
    sig = Dense(5, activation='sigmoid', name='fc/sigmoid')(m.layers[-2].output)
    m = Model(m.input, [sof, sig])
    m.compile(optimizer='adam', loss={'fc/softmax': 'categorical_crossentropy', 'fc/sigmoid': 'binary_crossentropy'},
              metrics=['accuracy'])

    for layer in m.layers:
        layer.trainable = True

    if not gpu_avaliable:
        batch_size = 256

    print(m.summary())
    start = time.perf_counter()

    m.fit(x_train, y_train,
          epochs=target_epoches,
          batch_size=batch_size,
          validation_split=0.3,
          shuffle=True,
          callbacks=training_callback(results_folder=results_folder))

    time_spend = time.perf_counter() - start
    m.save('{}/{}.model.h5'.format(results_folder, "squeezenet"))
    print("Training Spend: ", time_spend/60)
    print("Save in folder: ", results_folder)
    model_evaluate(model_path=results_folder, x_test=x_test, y_test=y_test)


def predicting_video_segmentation(model_path, img_path, results_folder):
    custom_obj = None
    classes = load_class_labels(model_path=model_path, results_folder=results_folder)
    if not os.path.exists(img_path):
        raise ValueError("File: {} does not exist.".format(img_path))

    # Load model
    pred = load_model(model_path, custom_objects=custom_obj)

    # Load image files
    print("Loading image files...")
    if os.path.isdir(img_path):
        img_path = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        img_path = [i for i in img_path if os.path.isfile(i)]

    print("Len: {}, Predicting...".format(len(img_path)))
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
    else:
        start = time.perf_counter()
        for i in img_path:
            try:
                i_array = image.img_to_array(image.load_img(i, target_size=target_size))
                i_array = i_array.reshape(1, *i_array.shape)
                # Predict
                rs = pred.predict(i_array)
                pd_rs.append([rs[0][0], rs[1][0]])

            except Exception:
                print("Remove: ", i)
                os.remove(i)
        time_spend = time.perf_counter() - start
        pd_rs = np.array(pd_rs)

    rs = []
    for key, i in enumerate(img_path):
        try:
            softmax = classes[0][int(np.argmax(pd_rs[key][0]))]
            softmax_prob = np.max(pd_rs[key][0]) * 100
            sigmoid = classes[1][int(np.argmax(pd_rs[key][1]))] if np.max(pd_rs[key][1]) > 0.2 else ""
            sigmoid_prob = np.max(pd_rs[key][1]) * 100
            rs.append({'key': int(i.split('.')[-2]),
                       'path': i,
                       'filename': os.path.basename(i),
                       'status':  "{}({:.2f}%)/{}({:.2f}%)".format(softmax, softmax_prob, sigmoid, sigmoid_prob),
                       'detail': ", ".join(["%02f" % i for i in np.concatenate(pd_rs[key])])})
        except Exception:
            import pytest; pytest.set_trace()

    for key, i in enumerate(sorted(rs, key=lambda x: x['key'])):
        if key % 20 == 0:
            print('\t', '\t', '\t', '\t', '\t', classes)
        print("%04d" % i['key'], '\t', i['filename'][-20:], '\t', "%08s" % i['status'], '\t', i['detail'])
    print(pred_num, "rows", "\t", time_spend, "sec.", "\t", pred_num // time_spend, "fps.", "\t", arrow.now())
    print()
    return img_path, pd_rs, rs


def model_evaluate(model_path, x_test, y_test):
    if os.path.isfile(model_path):
        models = [model_path]
    else:
        models = [os.path.join(model_path, i) for i in sorted(os.listdir(model_path)) if '.h5' in i]
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


@cbox.cmd
def main(operation='', path='', model_path='', classname="", pickup_mode="copy"):
    results_folder = os.path.join("results", arrow.now().format("YYYY-MM-DD-HH-mm-ss") + "-" + operation)
    os.makedirs(results_folder, exist_ok=True)
    [os.rmdir(os.path.join("results", i)) for i in os.listdir("results")
     if os.path.join("results", i) != results_folder and os.path.isdir(i) and
     len(os.listdir(os.path.join("results", i))) == 0]

    if operation == '':
        print("Operation List: ")
        print("train: --path, --use-model")
        print("predict: --model-path, --path")
        print("image-augmentation: --path")
        print("video-clip: --path")
        print("generate-prediction-result-to-video: --model-path, --path")
        print("pickup-prediction-result: --model-path, --path, --classname, --pickup-mode")

    if operation == 'image-augmentation':
        data_augmentation(folder=path, gen_pics=1, results_folder=results_folder)

    if operation == 'train':
        if not os.path.exists(path):
            assert ValueError, "Dataset folder is not exist: " + path
        X, y, _ = load_dataset(dataset_folder=path, results_folder=results_folder, img_preload=True, target_size=target_size)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
        training_video_segmentation(x_train, x_test, y_train, y_test, results_folder=results_folder)

    if operation == 'predict':
        predicting_video_segmentation(model_path=model_path, img_path=path, results_folder=results_folder)
        os.rmdir(results_folder)

    if operation == 'pickup-prediction-result':
        classes = load_class_labels(model_path=model_path, results_folder=results_folder)
        if classname not in classes:
            assert ValueError, "Class name isn't exist! Available: " + ", ".join(classes)
        cls_index = classes.index(classname)
        img_path, pred, _ = predicting_video_segmentation(img_path=path,
                                                          model_path=model_path, results_folder=results_folder)
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
        print("Result save in: ", results_folder)

    if operation == 'generate-prediction-result-to-video':
        if not model_path or not path:
            assert ValueError, "Require --model-path & --path"
        if os.path.isfile(path):
            basename = os.path.basename(path)[:-4]
            clip = VideoFileClip(path)
            clip.write_images_sequence(nameformat="{}/{}.frame.%05d.jpg".format(results_folder, basename), fps=5)
            # for smaller file size.
            for p in os.listdir(results_folder):
                if '.jpg' in p:
                    img = Image.open(os.path.join(results_folder, p))
                    img = img.resize((int(img.size[0] * 0.7), int(img.size[1] * 0.7)), Image.ANTIALIAS)
                    img.save(os.path.join(results_folder, p))
            path = results_folder

        print()
        # classes = load_class_labels(model_path=model_path, results_folder=results_folder)
        # color_index = np.random.choice(range(150, 255, 5), (max([len(i) for i in classes]), 3), replace=False)
        font = ImageFont.truetype("OpenSans-Semibold.ttf", 30)

        img_path, pred, rs = predicting_video_segmentation(img_path=path, model_path=model_path,
                                                           results_folder=results_folder)

        for key, p in enumerate(sorted(rs, key=lambda x: x['key'])):
            # cls_index_softmax = int(np.argmax(pred[key][0]))
            # cls_index_sigmoid = int(np.argmax(pred[key][1]))
            # s_softmax = classes[0][cls_index_softmax] + " : " + str(np.max(pred[key][0]))
            # s_sigmoid = classes[1][cls_index_sigmoid] + " : " + str(np.max(pred[key][1])) + " / " +
            # str((pred[key][1]*100).astype(int))
            img = Image.open(p['path'])
            draw = ImageDraw.Draw(img)
            status = p['status'].split('/')
            draw.text((20, 10), text=status[0], fill=(255, 150, 150), font=font)
            draw.text((20, 50), text=status[1], fill=(150, 255, 150), font=font)
            # if np.max(pred[key][1]) > 0.3:
            #     draw.text((10, 40), text=s_sigmoid, fill=tuple(color_index[cls_index_sigmoid]), font=font)
            img.save(p['path'])

            if key % 1000 == 0:
                print("Proccessing: {}/{}.".format(key, len(img_path)))

        filename = os.path.basename(img_path[0][:-10]) + ".mp4"
        clip = ImageSequenceClip(path, fps=10)
        clip.write_videofile(os.path.join(results_folder, filename), codec='libx264', threads=4)
        [os.remove(i) for i in img_path]

    if operation == 'video-clip':
        if not os.path.isfile(path):
            assert ValueError, "Video File doesn't exist. " + path
        basename = os.path.basename(path)[:-4]
        clip = VideoFileClip(path)
        clip.write_images_sequence(nameformat="{}/{}.frame.%05d.jpg".format(results_folder, basename), fps=1)

    K.clear_session()


if __name__ == '__main__':
    cbox.main(main)
