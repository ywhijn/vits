import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import editdistance
from tqdm import tqdm
from normalizers import EnglishTextNormalizer
# from normalizers import EnglishTextNormalizer
from datetime import datetime
import torch.multiprocessing as mp

import warnings


warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper.generation_whisper")

def calculate_wer(reference, hypothesis):
    """计算词错误率 (WER)"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words)


def process_directory_batch(pipe, directory, batch_size=16):
    """以批处理方式处理单个目录中的所有音频文件"""
    wer_sum = 0
    file_count = 0
    normalizer = EnglishTextNormalizer()
    # 读取文本文件
    text_file = os.path.join(directory, "id_text.txt")
    with open(text_file, 'r') as f:
        text_lines = f.readlines()

    text_dict = {line.split()[0]: ' '.join(line.split()[1:]) for line in text_lines}

    wav_files = [f for f in os.listdir(directory) if f.endswith('_output.wav')]

    print(f"whisper processing {directory}")
    for i in range(0, len(wav_files), batch_size):
        batch_files = wav_files[i:i + batch_size]
        batch_paths = [os.path.join(directory, f) for f in batch_files]
        batch_ids = [f.split('_')[0] for f in batch_files]

        original_texts = [text_dict.get(file_id, "").strip() for file_id in batch_ids]

        valid_indices = [i for i, text in enumerate(original_texts) if text]
        if not valid_indices:
            continue

        batch_paths = [batch_paths[i] for i in valid_indices]
        original_texts = [original_texts[i] for i in valid_indices]

        results = pipe(batch_paths)
        # recognized_texts = [result["text"].strip() for result in results]

        for i, res in enumerate(results):
            results[i]['normalized_text'] = normalizer(res['text'].strip())
        normalized_original_texts = [normalizer(text.strip().lower()) for text in original_texts]
        wers = [calculate_wer(orig, recog['normalized_text'])
                for orig, recog in zip(normalized_original_texts, results)]
        wer_sum += sum(wers)
        file_count += len(wers)
    return round(wer_sum * 100, 3), file_count


def process_directory(pipe, directory):
    """处理单个目录中的所有音频文件"""
    wer_sum = 0
    file_count = 0

    # 读取文本文件
    text_file = os.path.join(directory, "id_text.txt")
    with open(text_file, 'r') as f:
        text_lines = f.readlines()

    text_dict = {line.split()[0]: ' '.join(line.split()[1:]) for line in text_lines}

    # 获取目录中所有的 wav 文件
    wav_files = [f for f in os.listdir(directory) if f.endswith('_output.wav')]

    for wav_file in tqdm(wav_files, desc=f"Processing {directory}"):
        file_id = wav_file.split('_')[0]
        wav_path = os.path.join(directory, wav_file)

        # 获取原始文本
        original_text = text_dict.get(file_id, "").strip()

        if not original_text:
            print(f"Warning: No text found for audio file {wav_file}")
            continue

        # 使用 Whisper 进行语音识别
        result = pipe(wav_path)
        recognized_text = result["text"].strip().upper()

        # 计算 WER
        wer = calculate_wer(original_text, recognized_text)
        wer_sum += wer
        file_count += 1
    return wer_sum, file_count


# class ExperimentLogger:
#     def __init__(self, file_path='whisper_experiment_log.csv'):
#         self.file_path = file_path
#         self.columns = ['exp_name', 'model_id', 'dev-clean', 'dev-other', 'test-clean', 'test-other', 'average_wer', 'time', 'total_time']

#         if os.path.exists(self.file_path):
#             self.df = pd.read_csv(self.file_path)
#         else:
#             self.df = pd.DataFrame(columns=self.columns)

#     def add_record(self, exp_name, results):
#         record = {
#             'exp_name': exp_name,
#             'model_id': results['model_id'],
#             'dev-clean': results['dev-clean'],
#             'dev-other': results['dev-other'],
#             'test-clean': results['test-clean'],
#             'test-other': results['test-other'],
#             'average_wer': results['average_wer'],
#             'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             'total_time': results['total_time']
#         }

#         new_record = pd.DataFrame([record])
#         self.df = pd.concat([self.df, new_record], ignore_index=True)
#         self.save()

#     def save(self):
#         self.df.to_csv(self.file_path, index=False)

#     def display(self):
#         print(self.df)
import csv
import os
from datetime import datetime


class ExperimentLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.columns = ['exp_name', 'model_id', 'dev-clean', 'dev-other', 'test-clean', 'test-other', 'average_wer',
                        'time', 'total_time']

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()

    def add_record(self, results):
        record = {
            'exp_name': results['exp_name'],
            'model_id': results['model_id'],
            'dev-clean': results['dev-clean'],
            'dev-other': results['dev-other'],
            'test-clean': results['test-clean'],
            'test-other': results['test-other'],
            'average_wer': results['average_wer'],
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_time': results['total_time']
        }

        with open(self.file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(record)

    def display(self):
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(row)


# def process_file(pipe, audio_file, reference_file):
#     with open(reference_file, 'r') as f:
#         reference = f.read().strip()

#     result = pipe(audio_file)
#     prediction = result["text"]
#     wer = jiwer.wer(reference, prediction)
#     return wer
def process_file(pipe, audio_file, reference_file):
    with open(reference_file, 'r') as f:
        reference = f.read().strip()
    result = pipe(audio_file)
    prediction = result["text"]
    ref_words = reference.split()
    pred_words = prediction.split()
    edit_dist = editdistance.eval(ref_words, pred_words)
    wer = edit_dist / len(ref_words) if len(ref_words) > 0 else 0
    return wer


# def process_directory(pipe, directory):
#     wer_sum = 0
#     file_count = 0
#     reference_file = os.path.join(root, file[:-4] + '.txt')
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.wav') and 'output' in file:
#                 audio_file = os.path.join(root, file)

#                 if os.path.exists(reference_file):
#                     wer = process_file(pipe, audio_file, reference_file)
#                     wer_sum += wer
#                     file_count += 1
#     return wer_sum, file_count

def process_test_set(gpu_id, model_id, test_set, data_dir, torch_dtype):
    num_gpus = torch.cuda.device_count()
    if num_gpus>1:
       gpu_id = (gpu_id+4) % num_gpus

    device = f"cuda:{gpu_id}"
    print(device, model_id)
    # model_id = "openai/whisper-large-v3"
    model_id="/home/ma-user/work/daxintan/llama_omni_model/whisper-large-v3_hf/"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=64,
    )

    directory = f"{data_dir}/{test_set}"
    wer_sum, file_count = process_directory_batch(pipe, directory)
    return test_set, wer_sum, file_count





if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "/home/ma-user/work/yangwenhan/whisper/large_v3"
    # model_id="openai/whisper-large-v3"
    # model = AutoModelForSpeechSeq2Seq.from_pretrained(
    #     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    # model.to(device)
    data_dir = "/home/ma-user/work/yangwenhan/data/speech_reconstruction"

    res_path = "/home/ma-user/work/yangwenhan/trainlog/u2s_res.csv"
    # exp_name =f"u2s_stu_100hFTbpeUnit"
    exp_name = f"LS100_offlineFTphone_ctc60epoch_fsqUnits/stu_units_Epoch220"
    data_dir = f"{data_dir}/{exp_name}"
    # run_whisper_evaluation(exp_name, model_id, data_dir, res_path=res_path)
    # processor = AutoProcessor.from_pretrained(model_id)
    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model,
    #     tokenizer=processor.tokenizer,
    #     feature_extractor=processor.feature_extractor,
    #     torch_dtype=torch_dtype,
    #     device=device,
    #     return_timestamps=True,
    #     max_new_tokens=128,
    #     chunk_length_s=30,
    #     batch_size=16,
    # )
    # # 定义测试集

    # total_wer = 0
    # total_files = 0
    # data_dir = "/home/ma-user/work/yangwenhan/data/speech_reconstruction"
    # for test_set in test_sets:
    #     directory = f"{data_dir}/u2s_100hFTbpeUnit/{test_set}"
    #     wer_sum, file_count = process_directory(directory)
    #     total_wer += wer_sum
    #     total_files += file_count

    #     print(f"{test_set} - Average WER: {wer_sum/file_count:.4f}") editdistance

    # # 打印总体 WER
    # print(f"Overall Average WER: {total_wer/total_files:.4f}")
# /home/ma-user/anaconda3/envs/python-3.9.11/bin/python  -m pip install torch transformers editdistance tqdm /home/ma-user/work/yangwenhan/whisper/evaluate.py
