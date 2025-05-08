
import logging
import math
import subprocess
from lhotse import CutSet
import os
from person.utils import str2bool, ExperimentLogger,ErtraExperimentLogger,AllExperimentLogger, ConfigManager, pre_process_units, get_parser,find_latest_checkpoint_num
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import editdistance
from tqdm import tqdm
from normalizers import EnglishTextNormalizer
# from normalizers import EnglishTextNormalizer
from datetime import datetime
import torch.multiprocessing as mp
import json
from my_train_ms_single_gpu import main

from synthesis_unit_sequence import run_parallel_synthesis
import psutil
import subprocess
import argparse

def check_for_checkpoint(output_model_path, target_ckpt="G_50000.pth"):
    return os.path.exists(os.path.join(output_model_path, target_ckpt))


def kill_train_processes():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'train_u2s' in ' '.join(proc.info['cmdline']):
            proc.terminate()

    # 等待进程终止
    time.sleep(5)

    # 如果进程仍然存在，强制结束
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'train_u2s' in ' '.join(proc.info['cmdline']):
            proc.kill()


def check_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
        return "No running processes found" not in result.stdout
    except:
        print("无法检查GPU使用情况。请确保已安装nvidia-smi。")
        return False


def wait_for_gpu_release():
    while check_gpu_usage():
        print("等待GPU资源释放...")
        time.sleep(60)  # 每分钟检查一次
    print("所有GPU资源已释放。")



from start import initialize


def make_validate_scratch(src_path, output_path):
    all_dataset = "LS960"
    parts = ["librispeech_cuts_dev-other.jsonl.gz"]

    os.makedirs(output_path, exist_ok=True)
    source_output_file = Path(output_path) / "dev-other_wav_sid_reduced_unit.txt"
    if os.path.exists(source_output_file):
        print(f"{source_output_file} already existed")
        return False
    audio_paths = []
    speaker_list = []
    units = []
    speaker2sid_file = Path(output_path) / (all_dataset + '_speaker2sid.txt')
    if os.path.exists(speaker2sid_file):
        with open(speaker2sid_file, 'r') as f:
            speaker2sid = f.readlines()
            speaker2sid = {line.strip().split()[0]: int(line.strip().split()[1]) for line in speaker2sid}
    for part in parts:
        cuts_dir = f"{src_path}/{part}"
        cut_set = CutSet.from_file(cuts_dir)

        for cut in cut_set:
            for supervision, source in zip(cut.supervisions, cut.recording.sources):
                audio_paths.append(source.source)
                speaker_list.append(f"LS960_{supervision.speaker}")
                units.append(pre_process_units(supervision.custom['unit']))

    s2u_lines = []
    for wav_file, speaker, unit_sequence in zip(audio_paths, speaker_list, units):
        if speaker not in speaker2sid:
            speaker2sid[speaker]=len(speaker2sid)
        speaker_id = speaker2sid[speaker]
        new_line = '|'.join([wav_file, str(speaker_id), " ".join(unit_sequence)])
        s2u_lines.append(new_line)
    with open(source_output_file, 'w') as f:
        f.write('\n'.join(s2u_lines))
    print(f'data is save in {source_output_file}')
def make_data_scratch(src_path, output_path):
    all_dataset = "LS960"
    parts=["librispeech_cuts_train-other-500.jsonl.gz","librispeech_cuts_train-clean-360.jsonl.gz","librispeech_cuts_train-clean-100.jsonl.gz" ]


    os.makedirs(output_path,exist_ok=True)
    source_output_file = Path(output_path) / "wav_sid_reduced_unit.txt"
    # if os.path.exists(source_output_file):
    #     print(f"{source_output_file} already existed" )
    #     return False
    audio_paths= []
    speaker_list = []
    units = []
    # speaker_id_file="/home/ma-user/work/daxintan/data_for_fairseq_speech/for_U2S_training/40ms_yangwenhan_continue_pretrain_600k_data2vec_large_FSQ_8888_CTC_sampled_EN_CH_5000h_shanghai_GPU_mt_640k_20240911_8p/LibriTTS_WenetSpeech4TTS/train-clean-360_Premium/LibriTTS_WenetSpeech4TTS_speaker2sid.txt"
    # with open(speaker_id_file,'r') as f:
    #     speakers=f.readlines()

    for part in parts:
        cuts_dir = f"{src_path}/{part}"
        cut_set = CutSet.from_file(cuts_dir)

        for cut in cut_set:
            for supervision,source in zip(cut.supervisions,cut.recording.sources):
                audio_paths.append(source.source)
                speaker_list.append(f"LS960_{supervision.speaker}")
                units.append(pre_process_units(supervision.custom['unit']))

    speaker_set = sorted(list(set(speaker_list)))
    speaker2sid = {speaker: i for i, speaker in enumerate(speaker_set)}
    speaker2sid_file = Path(output_path) / (all_dataset + '_speaker2sid.txt')
    with open(speaker2sid_file, 'w') as f:
        for speaker in speaker_set:
            f.write(speaker + ' ' + str(speaker2sid[speaker]) + '\n')

    s2u_lines=[]
    for wav_file,speaker,unit_sequence in zip(audio_paths,speaker_list,units):
        speaker_id = speaker2sid[speaker]
        new_line = '|'.join([wav_file, str(speaker_id), " ".join(unit_sequence)])
        s2u_lines.append(new_line)
    with open(source_output_file, 'w') as f:
        f.write('\n'.join(s2u_lines))
    print(f'data is save in {source_output_file}')
    return len(speaker2sid)


    
def make_data_infer(src_path, output_path,test_sets= ["dev-clean", "dev-other", "test-clean", "test-other"]):
    all_dataset = "LS960"
    # parts=["librispeech_cuts_train-clean-100.jsonl.gz","librispeech_cuts_train-clean-360.jsonl.gz","librispeech_cuts_train-clean-100.jsonl.gz"]
     
    # train prepare
    dataset = 'LibriSpeech'
    language='English'
    def reformat_cuts(name):
        cuts_prefix = "librispeech_cuts_"
        cuts_post_fix = ".jsonl.gz"
        return cuts_prefix + name + cuts_post_fix

    os.makedirs(output_path,exist_ok=True)
    # speaker_id_file="/home/ma-user/work/daxintan/data_for_fairseq_speech/for_U2S_training/40ms_yangwenhan_continue_pretrain_600k_data2vec_large_FSQ_8888_CTC_sampled_EN_CH_5000h_shanghai_GPU_mt_640k_20240911_8p/LibriTTS_WenetSpeech4TTS/train-clean-360_Premium/LibriTTS_WenetSpeech4TTS_speaker2sid.txt"
    # with open(speaker_id_file,'r') as f:
    #     speakers=f.readlines()
    init_speaker_len=1e6
    speaker2sid={}
    speaker2sid_file = Path(output_path) / (all_dataset + '_speaker2sid.txt')
    if os.path.exists(speaker2sid_file):
        with open(speaker2sid_file, 'r') as f:
            speaker2sid = f.readlines()
            speaker2sid = {line.strip().split()[0]: int(line.strip().split()[1]) for line in speaker2sid}
        init_speaker_len=len(speaker2sid)
    for subset in test_sets:
        source_output_file = Path(output_path) / f"{subset}_reduced_unit_with_text.txt"
        if os.path.exists(source_output_file):
            print(f"{source_output_file} already existed")
            return False
        audio_paths = []
        speaker_list = []
        units = []
        texts= []
        part = reformat_cuts(subset)
        cuts_dir = f"{src_path}/{part}"
        cut_set = CutSet.from_file(cuts_dir)
        for cut in cut_set:
            for supervision,source in zip(cut.supervisions,cut.recording.sources):
                audio_paths.append(source.source)
                speaker_list.append(f"LS960_{supervision.speaker}")
                units.append(pre_process_units(supervision.custom['unit']))
                texts.append(supervision.text)
    # speaker_set = sorted(list(set(speaker_list)))
    # speaker2sid = {speaker: i for i, speaker in enumerate(speaker_set)}


        s2u_lines=[]
        for wav_file,text,unit_sequence in zip(audio_paths,texts,units):
            # if speaker not in speaker2sid:
            #     speaker2sid[speaker]=len(speaker2sid)
            # speaker_id = speaker2sid[speaker]
            new_line = '|'.join([wav_file, " ".join(unit_sequence), text ])
            s2u_lines.append(new_line)

        with open(source_output_file, 'w') as f:
            f.write('\n'.join(s2u_lines))
        print(f'data is save in {source_output_file}')
    if init_speaker_len < len(speaker2sid):
        print("more speaker in test set!")
from whisper_evaluate import process_test_set
def run_whisper_evaluation(exp_name, model_id, update_num, data_dir, test_sets, logger, torch_dtype=torch.float16):


    start_time = datetime.now()

    # test_sets = ["dev-clean"]
    # test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    num_gpus = torch.cuda.device_count()
    # assert num_gpus >= 4, "This script requires at least 4 GPUs"

    mp.set_start_method('spawn', force=True)

    process_args = [(i, model_id, test_set, data_dir, torch_dtype) for i, test_set in enumerate(test_sets)]

    with mp.Pool(num_gpus) as pool:
        results = pool.starmap(process_test_set, process_args)

    total_wer = 0
    total_files = 0
    wer_results = {}

    for test_set, wer_sum, file_count in results:
        wer = wer_sum / file_count if file_count > 0 else 0
        wer_results[test_set] = round(float(wer),2)
        total_wer += wer_sum
        total_files += file_count

    average_wer = total_wer / total_files if total_files > 0 else 0

    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    wer_res={}
    for test_set in test_sets:
         wer_res[test_set]=wer_results[test_set]
    results = {
        'exp_name': exp_name,
        'model_id': update_num,
        # 'dev-clean': wer_results['dev-clean'],
        # 'dev-other': wer_results['dev-other'],
        # 'test-clean': wer_results['test-clean'],
        # 'test-other': wer_results['test-other'],
        'average_wer': average_wer,
        'total_time': total_time
    }
    wer_res.update(results)
    logger.add_record(wer_res)
    logger.display()

def train_u2s(args, config_path, output_model_path):
    argv = [
        f'--config={config_path}',
        f'--model={output_model_path}'
    ]
    # print(argv)
    # main(argv,args=args)

    initialize(argv,args=args)



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    is_stu = False
    stu_exp = None
    if "stu_units" in args.cuts_path: # named as common dir / teacher exp / stu_units_branch
        teacher_exp = str(args.cuts_path).split("/")[-2]
        stu_exp = str(args.cuts_path).split("/")[-1]
        unit_input_dir = os.path.join(args.input_path, teacher_exp, stu_exp)
        audio_output_dir = os.path.join(args.audio_output_dir, teacher_exp, stu_exp)
        exper_name=os.path.join(teacher_exp, stu_exp)
        is_stu = True
    else:
        teacher_exp = str(args.cuts_path).split("/")[-1]
        unit_input_dir = os.path.join(args.input_path, teacher_exp)
        audio_output_dir = os.path.join(args.audio_output_dir, teacher_exp)
        exper_name = teacher_exp

    os.makedirs(unit_input_dir, exist_ok=True)
    n_speakers = 2338
    if not is_stu:
        n_speakers = make_data_scratch(args.cuts_path, unit_input_dir)
        validate_data_path = make_validate_scratch(args.cuts_path, unit_input_dir)
    res_path=args.res_dir
    if args.extra_data is not None:
        if "only" in args.extra_data:
            test_sets=["chime3-enh"]
            logger = ErtraExperimentLogger(file_path=args.res_dir)
        elif "all" in args.extra_data:
            test_sets=["dev-clean", "dev-other", "test-clean", "test-other","chime3-enh"]
            logger = AllExperimentLogger(file_path=args.res_dir)
    else:
        test_sets= ["dev-clean", "dev-other", "test-clean", "test-other"]
        logger = ExperimentLogger(file_path=args.res_dir)
    make_data_infer(args.cuts_path, unit_input_dir,test_sets)
    print(f"n_speakers, {n_speakers}")

    output_model_path = os.path.join("/home/ma-user/work/yangwenhan/u2s/model/", teacher_exp) # only teacher trains model
    os.makedirs(args.config_dir , exist_ok=True)
    output_config_path =f"{args.config_dir}/{teacher_exp}.json"
    if not os.path.exists(output_config_path):
        if "large" in str(args.cuts_path):
            config_manager = ConfigManager("/home/ma-user/work/yangwenhan/u2s/person/u2s_large.json")
        else:
            config_manager = ConfigManager("/home/ma-user/work/yangwenhan/u2s/person/u2s.json")  # original congig
        print("update following params")
        config_manager.update_field("data", "training_files", os.path.join(unit_input_dir, "wav_sid_reduced_unit.txt" ) )
        config_manager.update_field("data", "validation_files", os.path.join(unit_input_dir, "dev-other_wav_sid_reduced_unit.txt" ) )
        config_manager.update_field("data", "unit_num", args.unit_num)
        # config_manager.update_field("data", "n_speakers", n_speakers)
        config_manager.update_field("train", "batch_size", args.batch_size)
        config_manager.save_config(output_config_path)
        print("Configuration updated and saved: ", output_config_path)

    if args.train and ( not is_stu ):

        # begin training
        train_u2s(args, output_config_path, output_model_path)
    update_num = 60000
    # if not args.train:
    if args.checkpoint_dir is not None:
        checkpoint_file = args.checkpoint_dir
    else:
        ckpt_name = "G-best.pth"
        update_num = find_latest_checkpoint_num(output_model_path)
        checkpoint_file = f"{output_model_path}/{ckpt_name}"
    print("load exp_model", output_model_path)
    if args.synthesis:
        print("synthesis to ", audio_output_dir)
        run_parallel_synthesis(args=args,config_file=output_config_path, checkpoint_file=checkpoint_file,
                            unit_input_dir=unit_input_dir, audio_output_dir=audio_output_dir,
                            exper_name=exper_name, mode='reconstruction', unit_type='40ms',test_sets=test_sets)
    model_id = args.model_id

    run_whisper_evaluation(exp_name=exper_name, model_id=model_id,update_num=update_num, data_dir=audio_output_dir, test_sets=test_sets,logger=logger )
