import os
import math
import cbox
import json
import arrow
import jieba
import requests
import tempfile
import subprocess
import numpy as np
from pathlib import Path
# from ckip import CkipSegmenter
from collections import Counter
from multiprocessing import Process
from shutil import move, rmtree, copyfileobj
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


streaming_folder = 'soocii-dataset/streaming_files'
user_mapping_table_file = 'soocii-dataset/user_mapping_table.json'
javis_streaming_records_file = 'soocii-dataset/javis_streaming_records.json'
firebase_streaming_comments_file = 'soocii-dataset/firebase_streaming_comments.json'
pepper_streaming_records_file = "soocii-dataset/pepper_streaming_records.json"
combined_dataset_file = 'soocii-dataset/combined_streaming_comments_records.json'
downloaded_video_records_file = "soocii-dataset/downloaded_video_records.json"


def download_file(uri, save_path):
    try:
        print("Downloading {} ...".format(save_path.split('/')[-1]))
        response = requests.get(uri, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                copyfileobj(response.raw, f)
            del response
        else:
            raise Exception("Access Denied, %s" % uri)
    except Exception as e:
        print("Error: ", e)


def download_video_batch(data):
    f_name = []
    saved_video = []
    if not os.path.exists(streaming_folder):
        os.mkdir(streaming_folder)
    if not os.path.exists(downloaded_video_records_file):
        with open(downloaded_video_records_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps([]))
    else:
        saved_video = json.load(open(downloaded_video_records_file, 'r', encoding='utf-8'))

    for i in data:
        tmp_name = os.path.join(streaming_folder, i['streaming_name'] + ".mp4")
        if tmp_name not in saved_video:
            if os.path.exists(tmp_name):
                saved_video.append(i['streaming_name'] + ".mp4")
            if arrow.get(i['start_at']) > arrow.now().shift(months=-1):
                f_name.append((i['streaming_url'], tmp_name))
            elif i['pepper_streaming_name']:
                f_name.append(("https://cdn-gcp.soocii.me/pepper-prod-gcp-asia-east1-streaming-recall/%s" % i['pepper_streaming_name'], tmp_name))
    if 3 > len(data):
        [download_file(uri=url, save_path=f) for url, f in f_name]
    else:
        jobs = [Process(target=download_file, kwargs={'uri': url, 'save_path': f}) for url, f in f_name]
        job_que = [jobs[i * 3:i * 3 + 3] for i in range(int(math.ceil(len(jobs) / 3)))]
        print(job_que)
        for que in job_que:
            [j.start() for j in que]
            [j.join() for j in que]

    saved_video += [os.path.basename(name) for uri, name in f_name]
    with open(downloaded_video_records_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(saved_video, ensure_ascii=False))


def update_user_mapping_table():
    with open(user_mapping_table_file, 'r', encoding='utf-8') as f:
        ret = json.load(f)
    with tempfile.TemporaryDirectory() as dir_name:
        download_dir = Path(dir_name)
        download_dir = download_dir / 'soocii_user'
        aws_cli = "aws s3 sync s3://jarvis-prod-backend-media/soocii_user {}".format(download_dir)
        subprocess.run(aws_cli, shell=True)
        for file_path in download_dir.glob('**/soocii_user'):
            with file_path.open() as f:
                for line in f:
                    data = json.loads(line)
                    ret[data['id']] = data['soocii_id']
    with open(user_mapping_table_file, 'w', encoding='utf-8') as f:
        json.dump(ret, f)


def format_pepper_streaming_records():
    pepper_records_tmp = 'soocii-dataset/pepper_tmp_records'
    os.system('aws s3 cp s3://soocii-table/soocii_pepper/posted_streaming_table.tsv {} --recursive'.format(
        pepper_records_tmp))
    daily_records = "\n".join([
        open(os.path.join(pepper_records_tmp, daypath, "posted_streaming_table.tsv"), encoding='utf-8').read()
        for daypath in os.listdir(pepper_records_tmp)])
    rs = {i.split('\t')[6]: i.split('\t')[0] for i in daily_records.split('\n') if i and i.split('\t')[6] != 'null'}
    with open(pepper_streaming_records_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(rs, sort_keys=True))
    rmtree(pepper_records_tmp)


def format_jarvis_streaming_records():
    javis_records_raw_tmp_folder = 'soocii-dataset/javis_tmp_records'
    javis_records_raw_path = 'soocii-dataset/javis_streaming_records'

    datelist = [arrow.now().shift(days=-i).format("YYYYMMDD") for i in range(60)]
    datelist = [i for i in datelist if i not in [i.split('_')[2] for i in os.listdir(javis_records_raw_path)]]

    for date in datelist:
        awsquery = "aws s3 cp s3://jarvis-prod-backend-media/soocii_streaming/day={} {}/day={} --recursive".format(date, javis_records_raw_tmp_folder, date)
        os.system(awsquery)
    format_pepper_streaming_records()
    p_records = json.load(open(pepper_streaming_records_file, encoding='utf-8'))

    if datelist:
        for day_folder in os.listdir(javis_records_raw_tmp_folder):
            base_path = os.path.join(javis_records_raw_tmp_folder, day_folder)
            ori_path = os.path.join(base_path, 'soocii_streaming')
            save_path = os.path.join(javis_records_raw_path, 'soocii_streaming_%s' % day_folder.split('=')[-1])
            if os.path.exists(ori_path):
                move(ori_path, save_path)
            else:
                if os.path.isdir(base_path):
                    data_str = "\n".join([
                        open(os.path.join(base_path, i), encoding='utf-8').read() for i in os.listdir(base_path)
                        if i != '.DS_Store'
                    ])
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(data_str)
        rmtree(javis_records_raw_tmp_folder)

    d = []
    for i in os.listdir(javis_records_raw_path):
        for j in open(javis_records_raw_path + "/" + i, encoding='utf-8').read().split('\n'):
            try:
                d.append(json.loads(j))
            except Exception as e:
                print("File Error:", i, " Exception:", e)
    d = [
        {"pepper_streaming_name": p_records[i['streaming_name']] if i.get('streaming_name') in p_records.keys() else "",
         **i} for i in d]
    data = json.dumps(d, ensure_ascii=False, sort_keys=True, indent=2)
    with open(javis_streaming_records_file, 'w', encoding='utf-8') as f:
        f.write(data)


def transfer_original_chatroom_2_useful():
    firebase_raw_path = 'soocii-dataset/firebase_streaming_comments'

    with open(firebase_streaming_comments_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = {i: data[i] for i in data if len(i.split('_')) > 2}
    for i in os.listdir(firebase_raw_path):
        if i[-5:] == '.json':
            print("Proccessing ", i)
            with open(firebase_raw_path + '/' + i, 'r', encoding='utf-8') as f:
                d = json.load(f)

            dd = d['release']['chat']
            for ts in dd.keys():
                if len(dd[ts]) > 1:
                    streaming_names = [i for i in dd[ts].keys() if i not in ['streaming_info', 'test'] and len(i) == 42]
                    for s_name in streaming_names:
                        data.update({"{}_{}".format(ts, s_name): [
                            dd[ts][s_name][row] for row in dd[ts][s_name].keys()
                            if dd[ts][s_name][row]['text'] not in ['has entered chat room', '已經進入聊天室',
                                                                   'チャットルームに参加しました', 'null']
                        ]})

    with open(firebase_streaming_comments_file, 'w', encoding='utf-8') as f:
        data = {i: data[i] for i in data if data[i]}
        f.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2))


def combine_message_records():
    rs = []
    javis_f = json.load(open(javis_streaming_records_file, 'r', encoding='utf-8'))
    javis_streaming_key = {i.get('streaming_name'): key for key, i in enumerate(javis_f)}
    firebase_f = json.load(open(firebase_streaming_comments_file, 'r', encoding='utf-8'))
    streaming_ids = filter(lambda x: x in javis_streaming_key.keys(), [i[11:] for i in firebase_f.keys()])

    for streaming_name in streaming_ids:
        javis_dataset = javis_f[javis_streaming_key[streaming_name]]
        start = javis_dataset.get('start_at')
        firebase_id = filter(lambda x: x[11:] == streaming_name, firebase_f.keys()).__next__()
        javis_dataset['chat'] = [{'time_location_min': (i.get('timestamp') // 1000 - start) / 60,
                                  'time_location_sec': (i.get('timestamp') // 1000 - start),
                                  **i} for i in firebase_f[firebase_id]]
        rs.append(javis_dataset)
    with open(combined_dataset_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(rs, indent=2, ensure_ascii=False, sort_keys=True))


def download_highrank_video(chat_per_min_gte, clip, duration):
    data = statistics_report(chat_per_min_gte, sort_by='game')
    download_video_batch(data)
    if clip:
        for i in data:
            try:
                clip_video(i['streaming_name'], clip_condition_by='fps', duration=duration)
            except:
                print("Clip {} failed.".format(i['streaming_name']))


def clip_video(video_id, clip_condition_by, duration):
    target_folder = ""
    video_file = os.path.join(streaming_folder, video_id+".mp4")
    video_info = [i for i in json.load(open(combined_dataset_file, 'r', encoding='utf-8')) if i['streaming_name'] == video_id][0]
    if not os.path.exists(video_file):
        download_video_batch([video_info])
        assert os.path.exists(video_file), "File Does not Exist."

    if clip_condition_by == 'fps':
        target_folder = os.path.join(streaming_folder, video_id + '_fps')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        clip = VideoFileClip(video_file)
        clip.write_images_sequence(nameformat="{}/{}.frame.%05d.jpg".format(target_folder, video_id), fps=1/duration)
    elif clip_condition_by == 'chat':
        target_folder = os.path.join(streaming_folder, video_id+'_clip_chat_avg_avg')
        video_length = video_info['end_at'] - video_info['start_at']
        chat_avg_count = len(video_info['chat']) / (video_length / duration)
        chat_time_list = [i['time_location_sec'] for i in video_info['chat']]

        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        rs = []
        previous_end = 0
        for key, row in enumerate(chat_time_list):
            tmp_count = 0
            tmp_list = []
            for sub_key, t in enumerate(chat_time_list[key:]):
                if row + duration >= t >= row:
                    tmp_count += 1
                    tmp_list.append(sub_key+key)
                else:
                    break
            if tmp_count > chat_avg_count * 2:
                current_end = row + duration
                if previous_end > row:
                    previous_layer = len(rs) - 1
                    print(key, "start:", row, "previous end recorded:", rs[previous_layer]['end'], "previous end:", previous_end)
                    assert rs[previous_layer]['end'] == previous_end
                    new_key = [i for i in tmp_list if i not in rs[previous_layer]['key']]
                    rs[previous_layer]['key'] += new_key
                    rs[previous_layer]['count'] = len(rs[previous_layer]['key'])
                    rs[previous_layer]['end'] = current_end
                else:
                    rs.append({'start': row, 'end': current_end, 'count': tmp_count, 'key': tmp_list})
                previous_end = current_end
        cut_avg = sum([i['count'] for i in rs]) / len(rs)
        [ffmpeg_extract_subclip(video_file, i['start']-5, i['end'],
                                targetname=os.path.join(target_folder, "{}-{}-{}-{}.mp4".format(
                                    i['start'], i['end'], i['end'] - i['start'], i['count'])))
         for key, i in enumerate(rs) if i['count'] >= cut_avg]
    with open(os.path.join(target_folder, video_id+".json"), 'w', encoding='utf-8') as f:
        f.write(json.dumps(video_info, ensure_ascii=False, indent=2))


def statistics_report(chat_per_min_gte, sort_by):
    rs = []
    combined_data = json.load(open(combined_dataset_file, 'r', encoding='utf-8'))
    user_mapping = json.load(open(user_mapping_table_file, 'r', encoding='utf-8'))
    for i in sorted(combined_data, key=lambda x: x[sort_by]):
        avg_chat_per_min = len(i['chat']) / ((i['end_at'] - i['start_at']) / 60)
        if (i['end_at'] - i['start_at']) > 300 and avg_chat_per_min > chat_per_min_gte:
            print("{}\t{}\t{:20}\t{:15}({})\t{:.4f}\t{}".format(arrow.get(i['start_at']).date(), i['streaming_name'], i['game'][:15], user_mapping.get(i['owner'], ''), i['owner'], avg_chat_per_min, len(i['chat'])))
            rs.append({'streaming_name': i['streaming_name'],
                       'streaming_url': i['streaming_url'],
                       'pepper_streaming_name': i['pepper_streaming_name'],
                       'start_at': i['start_at']})

    chat_per_min = [len(i['chat']) / ((i['end_at'] - i['start_at']) / 60) for i in combined_data]
    print("Total: ", len(combined_data))
    print("Chat Per Min: ", len([i for i in chat_per_min if i > 1]))
    print("Average: ", sum(chat_per_min) / len(combined_data))
    print("Max: ", np.max(chat_per_min))
    print("Min: ", np.min(chat_per_min))
    print("Std: ", np.std(chat_per_min))
    return rs


def statistics_chatroom_word_count(user, n_gram):
    rs = {}
    terms = []
    combined_data = json.load(open(combined_dataset_file, 'r', encoding='utf-8'))
    posts = [i['chat'] for i in combined_data if i['owner'] == user]
    sentences = [k['text'] for i in posts for k in i]
    print("Jieba: ")
    sent_seg = [term for key, sent in enumerate(sentences) for term in jieba.cut(sent)]
    terms += sent_seg
    for n in range(2, n_gram+1):
        terms += ["".join(set(sent[i:i + n])) for sent in sent_seg for i in range(0, len(sent) - n + 1)]
    counter = Counter(terms)
    value = list(map(lambda x: {x: counter[x]}, list(filter(lambda x: counter[x] > len(sentences)*0.005, counter))))
    [rs.update(v) for v in value]
    print("Total Sentences: ", len(sentences))
    print(sorted(rs.items(), key=lambda x: x[1], reverse=True))


    """
    seg = CkipSegmenter()
    print("Ckip: ")
    terms = []
    for key, i in enumerate(sentences):
        try:
            for k in seg.seg(i).tok:
                terms += k
        except:
            print("CKIP Exception: ", i)
    terms += [terms[i:i + n_gram] for i in range(0, len(terms) - n_gram + 1)]
    counter = Counter(terms)
    print(list(map(lambda x: {x: counter[x]}, list(filter(lambda x: counter[x] > len(sentences) * 0.01, counter)))))
    """


def statistics_video_comments(video_id):
    interval = 10
    if not video_id:
        return
    combined_data = json.load(open(combined_dataset_file, 'r', encoding='utf-8'))
    info = [i for i in combined_data if i['streaming_name'] == video_id][0]
    duration = info['end_at'] - info['start_at']
    val = {"{:f}~{:f}".format(i/60, (i+interval)/60): len([1 for k in info['chat'] if i < k['time_location_sec'] < i + interval])
           for i in range(0, duration, interval)}
    print(info['streaming_url'])
    for v in val:
        print(v, '\t', val[v])
    print("average: {}".format(len(info['chat'])/(duration/interval)))


@cbox.cmd
def main(operation, chat_per_min_gte=2.0, sort_by='game', user=None, n_gram=2, video_id='', duration=10.0, clip=False, clip_condition_by='fps'):
    if not os.path.exists(combined_dataset_file) or \
                    arrow.now().date() > arrow.get(os.stat(combined_dataset_file).st_mtime).date():
        format_jarvis_streaming_records()
        transfer_original_chatroom_2_useful()
        combine_message_records()
        update_user_mapping_table()

    if operation == 'update':
        format_jarvis_streaming_records()
        transfer_original_chatroom_2_useful()
        combine_message_records()
        update_user_mapping_table()

    elif operation == 'download-highrank':
        download_highrank_video(chat_per_min_gte=chat_per_min_gte, clip=clip, duration=duration)

    elif operation == 'clip-video':
        clip_video(video_id=video_id, clip_condition_by=clip_condition_by, duration=duration)

    elif operation == 'statistics-report':
        statistics_report(chat_per_min_gte=chat_per_min_gte, sort_by=sort_by)

    elif operation == 'statistics-wordcount':
        statistics_chatroom_word_count(user=user, n_gram=n_gram)

    elif operation == 'statistics-video-comments':
        statistics_video_comments(video_id=video_id)


if __name__ == '__main__':
    cbox.main(main)
