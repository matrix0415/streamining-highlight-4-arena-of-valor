import os
import time
import cbox
import json
import arrow
import shutil
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from keras import backend as K
from keras.layers import Dense
from keras.models import Model, load_model
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

from libs.squeezenet import SqueezeNet
from libs.gcp_utils import select_game_streaming_info
from libs.data_utils import load_dataset, load_class_labels, training_callback, data_augmentation
from libs.media_utils import video_to_img_ffmpeg, img_to_video, resize_img, concatenate_videos_ffmpeg, video_subclip_ffmpeg, download_file

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


def predicting_video_segmentation(model_path, img_path, results_folder=None, model_obj=None):
    classes = load_class_labels(model_path=model_path, results_folder=results_folder)
    if not os.path.exists(img_path):
        raise ValueError("File: {} does not exist.".format(img_path))

    # Load model
    if not model_obj:
        pred = load_model(model_path)
    else:
        pred = model_obj

    # Load image files
    print("Loading image files...")
    if os.path.isdir(img_path):
        img_path = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        img_path = [i for i in img_path if os.path.isfile(i)]
        pred_num = len(img_path)
    else:
        img_path = [img_path]
        pred_num = 1

    print("Len: {}, Predicting...".format(len(img_path)))

    pd_rs = []
    start = time.perf_counter()
    for i in img_path:
        try:
            i_array = image.img_to_array(image.load_img(i, target_size=target_size))
        except Exception:
            print("Remove: ", i)
            os.remove(i)
        # Predict
        tmp_rs = pred.predict(i_array.reshape(1, *i_array.shape))
        pd_rs.append([tmp_rs[0][0], tmp_rs[1][0]])

    time_spend = time.perf_counter() - start
    pd_rs = np.array(pd_rs)

    print_rs = []
    for key, i in enumerate(img_path):
        try:
            softmax = classes[0][int(np.argmax(pd_rs[key][0]))]
            softmax_prob = np.max(pd_rs[key][0]) * 100
            softmax_status_str = "{}({:.2f}%)".format(softmax, softmax_prob)
            sigmoid = [classes[1][i[0]] for i in np.argwhere(pd_rs[key][1] > 0.2)]
            sigmoid_prob = pd_rs[key][1][np.where(pd_rs[key][1] > 0.2)] * 100
            sigmoid_status_str = "{}({})".format(", ".join(sigmoid),
                                                 ", ".join(["{:.2f}%".format(i) for i in sigmoid_prob]))
            print_rs.append({'key': int(i.split('.')[-2]),
                             'path': i,
                             'filename': os.path.basename(i),
                             'status': "{}/{}".format(softmax_status_str, sigmoid_status_str),
                             'detail': ", ".join(["%02f" % i for i in np.concatenate(pd_rs[key])]),
                             'softmax_cls': softmax,
                             'softmax_prob': softmax_prob,
                             'sigmoid_cls': sigmoid,
                             'sigmoid_prob': sigmoid_prob})
        except Exception as e:
            print("Exception: ", e)
            pytest.set_trace()

    for key, i in enumerate(sorted(print_rs, key=lambda x: x['key'])):
        if key % 20 == 0:
            print('\t', '\t', '\t', '\t', '\t', classes)
        print("%04d" % i['key'], '\t', i['filename'][-20:], '\t', "%08s" % i['status'], '\t', i['detail'])
    print(pred_num, "rows", "\t", time_spend, "sec.", "\t", pred_num // time_spend, "fps.", "\t", arrow.now())
    print()
    return img_path, pd_rs, print_rs


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
def main(operation='', path='', model_path='', classname="", pickup_mode="copy", starts_from="", ends_to=""):
    start = arrow.now()
    results_folder = os.path.join("results", operation + "-" + arrow.now().format("YYYYMMDD-HHmmss"))
    os.makedirs(results_folder, exist_ok=True)
    [os.rmdir(os.path.join("results", i)) for i in os.listdir("results")
     if os.path.join("results", i) != results_folder and os.path.isdir(i) and
     len(os.listdir(os.path.join("results", i))) == 0]

    if operation == '':
        print("Operation List: ")
        print("train: --path")
        print("predict: --model-path, --path")
        print("image-augmentation: --path")
        print("video-to-img: --path")
        print("generate-prediction-result-from-bq-to-json: --model-path, --starts-from")
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

    if operation == 'generate-prediction-result-from-bq-to-json':
        if not model_path:
            assert ValueError, "Require --model-path"

        data = []
        model = load_model(model_path)
        prediction_results = {'model': model_path,
                              'prediction_utctimestamp': arrow.utcnow().timestamp,
                              'results': []}

        if path and os.path.isfile(path):
            try:
                data = json.load(open(path, 'r', encoding='utf-8'))
            except json.decoder.JSONDecodeError:
                assert ValueError, "Path file just be a json file."
        else:
            starts_from = arrow.get(starts_from).shift(hours=-8).timestamp
            ends_to = arrow.get(ends_to).shift(hours=-8).timestamp
            path = os.path.join(results_folder, 'dataset_collections_{}_to_{}.json'.format(starts_from, ends_to))
            data = select_game_streaming_info(game_name="傳說對決",
                                              starts_from_timestamp=starts_from, ends_to_timestamp=ends_to)
            data = sorted(data, key=lambda x: x['streaming_start_at'])
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2))

        data = [i for i in sorted(data, key=lambda x: x['streaming_start_at']) if i.get('streaming_exist')]
        for key, d in enumerate(data):
            rs = {}
            fps = 1
            individual_path = os.path.join(results_folder, d['streaming_name'] + ".json")
            tmp_video_file = os.path.join(results_folder, d['streaming_name'])
            print("Processing %s/%s %s ..." % (key, len(data), d['streaming_name']))

            if d['prediction_results_status']:
                continue

            rs = d
            rs['exception'] = None
            # rs['streaming_exist'] = False
            rs['prediction_results'] = None
            rs['prediction_results_status'] = False
            d['prediction_results_status'] = False
            # resp = requests.get('https://api.soocii.me/graph/v1.2/streaming_recalls',
            #                     params={'streaming_names': d['streaming_name']},
            #                     headers={'Authorization': "Bearer JNdNJriJ620nvj9RsyItfLvWkQhqWsXBxfN_V3JYivkIsgLB0hHujJWVGABBZQbLpgEVTztqbxPE83SE8cbibAnFyS1ECJ0-_kLEaGKrwTheFO-eOqm5TCCDHFCSkgtA4St7VmCaWNuzfEGInBrQqTkPE6vMbAdosWKqS6pGbS0NM3JQh9PyiYyHeck7b7PN2JXnh-IhckCjyozbdaWk8URKKpM9BYs-eUsK-BJhf9xOmW-U1RnrRcNHm4lDuSytrFhe22kMj9_EMYwJuweUwLyHc1d0-UupH962JJPDJBmJIYJ4J8ZnCoHDCVV6I_yr"}).json()
            # if not resp['success']:
            #     continue

            # rs['streaming_status_id'] = resp['data'][0]['attached_status_id']
            # rs['streaming_recall_id'] = resp['data'][0]['streaming_recall_id']
            # rs['streaming_message_history_url'] = resp['data'][0]['message_history_url']
            # rs['streaming_url'] = resp['data'][0]['streaming_recall_url']
            download_resp = download_file(url=rs['streaming_url'], save_path=tmp_video_file)

            if not download_resp[0]:
                print(download_resp[1])
                continue

            try:
                img_split_folder = os.path.join(results_folder, "split_images_" + d['streaming_name'])
                os.makedirs(img_split_folder)
                video_to_img_ffmpeg(video_file=tmp_video_file, target_path=img_split_folder, fps=fps)
                _, _, pred = predicting_video_segmentation(model_path=model_path,
                                                           img_path=img_split_folder, model_obj=model)
                prediction_results['results'] = [{'key': i['key'], 'second': i['key'] / fps,
                                                  'probabilities': [float(i) for i in i['detail'].split(',')],
                                                  'sigmoid_cls': i['sigmoid_cls'],
                                                  'sigmoid_prob': i['sigmoid_prob'].tolist(),
                                                  'softmax_cls': i['softmax_cls'],
                                                  'softmax_prob': i['sigmoid_prob'].tolist(),
                                                  'prediction_status': i['status']} for i in
                                                 sorted(pred, key=lambda x: x['filename'])]
                rs['prediction_results'] = prediction_results
                rs['streaming_exist'] = True
                d['prediction_results_status'] = True
                os.remove(tmp_video_file)
                tmp_d = json.dumps(rs, sort_keys=True, ensure_ascii=False)
                with open(individual_path, 'w', encoding='utf-8') as f:
                    f.write(tmp_d)
                del rs
            except Exception as e:
                print("Exception: ", e)
                rs['exception'] = str(e)
            shutil.rmtree(img_split_folder)
            tmp_d = json.dumps(data, sort_keys=True, ensure_ascii=False)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(tmp_d)

    if operation == 'generate-prediction-result-to-video':
        if not model_path or not path:
            assert ValueError, "Require --model-path & --path"
        if os.path.isfile(path):
            video_to_img_ffmpeg(video_file=path, target_path=results_folder, fps=1)
            resize_img(target_path=results_folder)
            path = results_folder

        font = ImageFont.truetype("conf/OpenSans-Semibold.ttf", 30)
        img_path, pred, rs = predicting_video_segmentation(img_path=path, model_path=model_path,
                                                           results_folder=results_folder)

        for key, p in enumerate(sorted(rs, key=lambda x: x['key'])):
            img = Image.open(p['path'])
            draw = ImageDraw.Draw(img)
            status = p['status'].split('/')
            draw.text((20, 10), text=status[0], fill=(255, 150, 150), font=font)
            draw.text((20, 50), text=status[1], fill=(150, 255, 150), font=font)
            img.save(p['path'])

            if key % 1000 == 0:
                print("Proccessing: {}/{}.".format(key, len(img_path)))

        filename = os.path.join(results_folder, os.path.basename(img_path[0][:-10]) + ".mp4")
        img_to_video(img_folder=path, target_file=filename, fps=3)

    if operation == 'video-to-img':
        video_to_img_ffmpeg(video_file=path, target_path=results_folder, fps=1)

    if operation == 'export-highlight-moment':
        if not path or not model_path:
            assert ValueError, "Require --model-path & --path"
        video_file_name = os.path.basename(path)
        img_split_folder = os.path.join(results_folder, "split_images")
        video_sections_folder = os.path.join(results_folder, "video_sections")
        os.makedirs(img_split_folder)
        os.makedirs(video_sections_folder)
        video_to_img_ffmpeg(video_file=path, target_path=img_split_folder, fps=1)
        _, _, rs = predicting_video_segmentation(model_path=model_path, img_path=img_split_folder)
        rs = sorted(rs, key=lambda x: x['filename'])
        for i in rs:
            i['softmax_prob'] = i['softmax_prob'].tolist()
            i['sigmoid_prob'] = i['sigmoid_prob'].tolist()
        meta = {'video-path': video_file_name, 'prediction-results': rs}

        with open(os.path.join(results_folder, "meta.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(meta, sort_keys=True, ensure_ascii=False))

        data = [[sec, val['softmax_cls'] if val['softmax_prob'] > 80 else "", val['sigmoid_cls']]
                for sec, val in enumerate(rs)]
        ngram = 5
        section_results = [max([detail[0] for detail in data[i:i + ngram]])
                           for i in range(0, len(data))
                           if len(data[i:i + ngram]) == ngram and
                           (list(set([row[1] for row in data[i:i + ngram]])) == ['playing'] and
                           'kill' in [col for row in data[i:i + ngram] for col in row[2]]) or
                           (list(set([row[1] for row in data[i:i + ngram - 3]])) == ['end'] and
                           'mvp' in [col for row in data[i:i + ngram - 3] for col in row[2]])
                           ]
        section_results = [[max(i-1, 0), i] for i in section_results]
        section_results = [[i[0], max(i[1],
                                      max([k[1] for k in section_results[key:] if k and i[1] > k[0] > i[0]] or [0]))]
                           for key, i in enumerate(section_results)
                           if key == 0 or i[0] > section_results[key - 1][1]]

        with open(os.path.join(results_folder, "video_sections.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(section_results, sort_keys=True, ensure_ascii=False))

        [video_subclip_ffmpeg(filename=path, t1=val[0], t2=val[1],
                              targetname=os.path.join(video_sections_folder, video_file_name+"_%05d.mp4" % key),
                              reencode=True)
         for key, val in enumerate(section_results)]

        concatenate_videos_ffmpeg(video_folder=video_sections_folder,
                                  target_file=os.path.join(results_folder, video_file_name + "-highlight.mp4"))
        shutil.rmtree(img_split_folder)
        shutil.rmtree(video_sections_folder)

    K.clear_session()
    print("Spent: {} mins.".format((arrow.now()-start).seconds/60))


if __name__ == '__main__':
    cbox.main(main)
