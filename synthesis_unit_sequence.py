import os
import shutil
import random
import logging
from pathlib import Path

import torch
from scipy.io.wavfile import write
from tqdm import tqdm
import torch.multiprocessing as mp
import commons
import utils
import torchaudio

from models import SynthesizerTrn
from text import text_to_sequence
from person.utils import str2bool, ExperimentLogger, ConfigManager, pre_process_units, get_parser
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


def synthesis(args,model_config_file, model_checkpoint_file, groundtruth_wav_list, text_list, unit_seq_list,
              synthesis_result_dir, mode, unit_type, print_verbose_info=True, streaming=False):
    Path(synthesis_result_dir).mkdir(parents=True, exist_ok=True)
    # load model
    hps = utils.get_hparams_from_file(model_config_file)
    unit_num=args.unit_num
    print(f'unit_num {unit_num}')
    if int(unit_num) == 4096:
        from text.symbols import symbols_with_4096 as symbols
    elif int(unit_num) == 1000:
        from text.symbols import symbols_with_1000 as symbols

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(model_checkpoint_file, net_g, None)
    # U2S
    # reconstruction
    if mode == 'reconstruction':
        id_text_list = []
        sample_index = 0
        for (groundtruth_wav_file, input_text, output_unit_sequence) in zip(groundtruth_wav_list, text_list, unit_seq_list):
            audio_id= os.path.splitext(os.path.basename(groundtruth_wav_file))[0]
            sample_index += 1
            id_text_list.append(f'{str(sample_index)} {input_text}')
            unit_type = 'output'
            unit_sequence = output_unit_sequence.strip()
            if len(unit_sequence) == 0:
                print('sample output is empty, skip')
                print(f'unit_sequence: {unit_sequence}')
                continue
            if any([not x.isnumeric() for x in unit_sequence.split(' ')]):
                print('sample output contains non-numeric symbol, skip')
                print(f'unit_sequence: {unit_sequence}')
                continue
            if len(unit_sequence) == 0:
                continue
            if print_verbose_info:
                print(f'unit_sequence: {unit_sequence}')
            # synthesize speech
            with torch.no_grad():
                unit_sequence = text_to_sequence(unit_sequence, hps.data.text_cleaners)
                unit_sequence = torch.LongTensor(unit_sequence)
                unit_sequence = unit_sequence.cuda().unsqueeze(0)
                unit_lengths = torch.LongTensor([unit_sequence.size(1)]).cuda()
                if streaming:
                    audio = net_g.infer_stream(
                        unit_sequence, unit_lengths, sid=None, noise_scale=.667, noise_scale_w=0.8,
                        length_scale=1, spec=None, spec_lengths=None)
                else:
                    audio = net_g.infer(
                        unit_sequence, unit_lengths, sid=None, noise_scale=.667, noise_scale_w=0.8,
                        length_scale=1, spec=None, spec_lengths=None)[0][0, 0].data.cpu().float().numpy()

            reconstructed_wav_file = synthesis_result_dir / f'{sample_index}_{audio_id}_{unit_type}.wav'
            write(reconstructed_wav_file, hps.data.sampling_rate, audio)
            target_groundtruth_wav = synthesis_result_dir / f'{sample_index}_{audio_id}_groundtruth.wav'
            shutil.copy(groundtruth_wav_file, target_groundtruth_wav)
        id_text_file = synthesis_result_dir / f'id_text.txt'
        with open(id_text_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(id_text_list))
    print(f'synthesized sample is saved into {synthesis_result_dir}')


def shift_unit(unit_seq_list, shift_num=1024):
    shifted_unit_seq_list = []
    for unit_seq in unit_seq_list:
        unit_seq = unit_seq.strip().lstrip('<|speech_').rstrip('|>')
        unit_seq = unit_seq.replace('|><|speech_', ' ')
        unit_list = unit_seq.split(' ')
        shifted_unit_list = [str(int(x) - shift_num) for x in unit_list]
        shifted_unit_seq = ''.join([f'<|speech_{x}|>' for x in shifted_unit_list])
        shifted_unit_seq_list.append(shifted_unit_seq)
    return shifted_unit_seq_list


def get_wav_text_unit_list(unit_text_file):
    wav_text_unit_list = []
    with open(unit_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            wav_file, unit_seq, text = line.split('|', maxsplit=2)
            # unit_seq = unit_seq.replace('$<', '').replace('>$', '')
            wav_text_unit_list.append((wav_file, text, unit_seq))
    return wav_text_unit_list


def get_sample(CH_wav_text_unit_list):
    random.seed(1234)
    random.shuffle(CH_wav_text_unit_list)

    # CH_wav_text_unit_list = CH_wav_text_unit_list[:sample_num]
    CH_wav_list = [x[0] for x in CH_wav_text_unit_list]
    CH_text_list = [x[1] for x in CH_wav_text_unit_list]
    CH_unit_seq_list = [x[2] for x in CH_wav_text_unit_list]
    return CH_wav_list, CH_text_list, CH_unit_seq_list


def multi_synthesis(model_config_file, model_checkpoint_file, groundtruth_wav_list, text_list, unit_seq_list,
              synthesis_result_dir, mode, unit_type, print_verbose_info=True, streaming=False):
    Path(synthesis_result_dir).mkdir(parents=True, exist_ok=True)
    # load model
    hps = utils.get_hparams_from_file(model_config_file)

    from text.symbols import symbols_with_4096 as symbols

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    net_g.eval()
    utils.load_checkpoint(model_checkpoint_file, net_g, None)

    # U2S reconstruction
    if mode == 'reconstruction':
        id_text_list = []
        sample_index = 0
        for (groundtruth_wav_file, input_text, output_unit_sequence) in tqdm(
                zip(groundtruth_wav_list, text_list, unit_seq_list)):
            sample_index += 1
            id_text_list.append(f'{str(sample_index)} {input_text}')
            unit_type = 'output'
            unit_sequence = output_unit_sequence.strip()
            if len(unit_sequence) == 0 or any([not x.isnumeric() for x in unit_sequence.split(' ')]):
                continue

            # synthesize speech
            with torch.no_grad():
                unit_sequence = text_to_sequence(unit_sequence, hps.data.text_cleaners)
                unit_sequence = torch.LongTensor(unit_sequence).cuda().unsqueeze(0)
                unit_lengths = torch.LongTensor([unit_sequence.size(1)]).cuda()
                if streaming:
                    audio = net_g.infer_stream(
                        unit_sequence, unit_lengths, sid=None, noise_scale=.667, noise_scale_w=0.8,
                        length_scale=1, spec=None, spec_lengths=None)
                else:
                    audio = net_g.infer(
                        unit_sequence, unit_lengths, sid=None, noise_scale=.667, noise_scale_w=0.8,
                        length_scale=1, spec=None, spec_lengths=None)[0][0, 0].data.cpu().float().numpy()

            reconstructed_wav_file = synthesis_result_dir / f'{sample_index}_{unit_type}.wav'
            write(reconstructed_wav_file, hps.data.sampling_rate, audio)
            target_groundtruth_wav = synthesis_result_dir / f'{sample_index}_groundtruth.wav'
            shutil.copy(groundtruth_wav_file, target_groundtruth_wav)

        id_text_file = synthesis_result_dir / f'id_text.txt'
        with open(id_text_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(id_text_list))

    print(f'synthesized sample is saved into {synthesis_result_dir}')


def process_subset(gpu_id,args, config_file, checkpoint_file, unit_input_dir,audio_output_dir, subset, exper_name, mode, unit_type):
    torch.cuda.set_device(gpu_id+4)

    EN_unit_text_file = f"{unit_input_dir}/{subset}_reduced_unit_with_text.txt"
    EN_wav_text_unit_list = get_wav_text_unit_list(EN_unit_text_file)
    EN_wav_list, EN_text_list, EN_unit_seq_list = get_sample(EN_wav_text_unit_list)
    EN_synthesis_result_dir = Path(
        f"{audio_output_dir}/{subset}")
    os.makedirs(EN_synthesis_result_dir, exist_ok=True)

    synthesis(args,config_file, checkpoint_file, EN_wav_list, EN_text_list, EN_unit_seq_list,
              EN_synthesis_result_dir, mode, unit_type, print_verbose_info=False)


def run_parallel_synthesis(args,config_file, checkpoint_file, unit_input_dir,audio_output_dir, exper_name, mode, unit_type,test_sets):
    # test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    
    num_gpus = torch.cuda.device_count()
    # assert num_gpus >= 4, "This script requires at least 4 GPUs"

    mp.set_start_method('spawn', force=True)

    processes = []
    for i, subset in enumerate(test_sets):
        p = mp.Process(target=process_subset,
                       args=(i,args, config_file, checkpoint_file, unit_input_dir,audio_output_dir, subset, exper_name, mode, unit_type))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def process_st_subset(gpu_id,args, config_file, checkpoint_file, unit_input_dir,audio_output_dir, subset, exper_name):
    num_gpus = torch.cuda.device_count()
    if num_gpus>1:
        torch.cuda.set_device(gpu_id+4)
    else:
        torch.cuda.set_device(gpu_id)

    EN_unit_text_file = f"{unit_input_dir}/{subset}_reduced_unit_with_text.txt"
    EN_wav_text_unit_list = get_wav_text_unit_list(EN_unit_text_file)
    EN_wav_list, EN_text_list, EN_unit_seq_list = get_sample(EN_wav_text_unit_list)
    print(subset,len(EN_wav_list))
    EN_synthesis_result_dir = Path(
        f"{audio_output_dir}/{subset}")
    os.makedirs(EN_synthesis_result_dir, exist_ok=True)

    st_synthesis(args,config_file,  EN_wav_list, EN_text_list,  EN_synthesis_result_dir,)

def st_synthesis(args,model_config_file, groundtruth_wav_list, text_list,
              synthesis_result_dir):
    Path(synthesis_result_dir).mkdir(parents=True, exist_ok=True)
    # load model


    from speechtokenizer import SpeechTokenizer
    pt_p="/home/ma-user/work/yangwenhan/SpeechTokenizer/res"
    config_path = f'/{pt_p}/config.json'
    ckpt_path = f'/{pt_p}/SpeechTokenizer.pt'
    model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
    model.eval()
    model = model.cuda()


    id_text_list = []
    sample_index = 0
    for (groundtruth_wav_file, input_text) in zip(groundtruth_wav_list, text_list):
        audio_id= os.path.splitext(os.path.basename(groundtruth_wav_file))[0]
        sample_index += 1
        id_text_list.append(f'{str(sample_index)} {input_text}')
        unit_type = 'output'
        wav, sr = torchaudio.load(groundtruth_wav_file)
        if wav.shape[0] > 1:
            wav = wav[:1,:]
        if sr != model.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.unsqueeze(0).cuda()
        with torch.no_grad():
            codes = model.encode(wav) # codes: (n_q, B, T)
        audio = model.decode(codes).data.cpu().float().numpy()

        reconstructed_wav_file = synthesis_result_dir / f'{sample_index}_{audio_id}_{unit_type}.wav'
        write(reconstructed_wav_file,sr, audio)
        target_groundtruth_wav = synthesis_result_dir / f'{sample_index}_{audio_id}_groundtruth.wav'
        shutil.copy(groundtruth_wav_file, target_groundtruth_wav)
    id_text_file = synthesis_result_dir / f'id_text.txt'
    with open(id_text_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(id_text_list))
    print(f'synthesized sample is saved into {synthesis_result_dir}')
    
def run_parallel_st_synthesis(args,config_file, checkpoint_file, unit_input_dir,audio_output_dir, exper_name,test_sets):
    # test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    
    num_gpus = torch.cuda.device_count()
    # assert num_gpus >= 4, "This script requires at least 4 GPUs"
    arg_processes=[]
    mp.set_start_method('spawn', force=True)
    for i, subset in enumerate(test_sets):
        arg_processes.append((i,args, config_file, checkpoint_file, unit_input_dir,audio_output_dir, subset, exper_name))

    with mp.Pool(num_gpus) as pool:
        results = list(tqdm(
                pool.starmap(process_st_subset, arg_processes),
            total=len(arg_processes),
            desc="Processing dataset parts"
        ))
    
from whisper_evaluate import process_test_set
from datetime import datetime
def run_whisper_evaluation(exp_name,data_dir, test_sets, logger, torch_dtype=torch.float16):


    start_time = datetime.now()

    # test_sets = ["dev-clean"]
    # test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    num_gpus = torch.cuda.device_count()
    # assert num_gpus >= 4, "This script requires at least 4 GPUs"

    mp.set_start_method('spawn', force=True)

    process_args = [(i, "0", test_set, data_dir, torch_dtype) for i, test_set in enumerate(test_sets)]

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
        'model_id': 0,
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
if __name__ == '__main__':
    mode = 'reconstruction'
    unit_type = '40ms'  # ['40ms', '40ms_L2_BN','40ms_L2_BN_multilingual'，'40ms_multilingual_8888','data2vec_40ms_multilingual_8888','data2vec_80ms_multilingual_86868']
    # LibriSpeech, AISHELL2: in-domain test data
    EN_dataset = 'LibriSpeech'
    config_file = "/home/ma-user/work/yangwenhan/u2s/person/u2s.json"
    parser = get_parser()
    args = parser.parse_args()
    is_stu = False
    if "stu_units" in args.cuts_path: # named as /teacher exp /stu_units_branch
        teacher_exp = str(args.cuts_path).split("/")[-2]
        input_path = os.path.join(args.input_path, teacher_exp) + str(args.cuts_path).split("/")[-1]
        is_stu = True
    else:
        teacher_exp = str(args.cuts_path).split("/")[-1]
        input_path = os.path.join(args.input_path, teacher_exp)
    unit_input_dir = "/home/ma-user/work/yangwenhan/u2s/data/streamPosDucerFsq4096_PhoneLS960_640ms30"
    audio_output_dir = "/home/ma-user/work/yangwenhan/data/speech_reconstruction"
    ckpt_dir = "/home/ma-user/work/yangwenhan/u2s/model/UnitS_BS28_hubert_8v100_finetune_debug_fsq_ctc_LS960_10.7"
    ckpt_name = "G-best.pth"
    exper_name = "100hFTbpeUnit"
    # # GigaSpeech, WenetSpeech
    # EN_dataset, EN_subset = 'GigaSpeech', 'test'
    # CH_dataset, CH_subset = 'WenetSpeech', 'test_meeting'
    is_stu = False
    # unit_input_dir = "/home/ma-user/work/yangwenhan/u2s/data/stu_units/LS100FTbpe_ctc220epoch_fsqUnits"
    if is_stu:
        exper_name = f"stu_{exper_name}"
    test_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    # test_sets = ["dev-clean"]


    checkpoint_file = f"{ckpt_dir}/{ckpt_name}"
    # is_stu = False
    exper_name="test"
    audio_output_dir = os.path.join(args.audio_output_dir, exper_name)
    # run_parallel_st_synthesis(args, config_file, checkpoint_file, unit_input_dir, audio_output_dir, exper_name, test_sets)
    logger = ExperimentLogger(file_path=args.res_dir)
    run_whisper_evaluation(exp_name=exper_name, data_dir=audio_output_dir, test_sets=test_sets,logger=logger )

    # for subset in test_sets:
    #     EN_unit_text_file = f"{unit_input_dir}/{subset}_reduced_unit_with_text.txt"
    #     EN_wav_text_unit_list = get_wav_text_unit_list(EN_unit_text_file)
    #     EN_wav_list, EN_text_list, EN_unit_seq_list = get_sample(EN_wav_text_unit_list)
    #     EN_synthesis_result_dir = Path(f"/home/ma-user/work/yangwenhan/data/speech_reconstruction/u2s_{exper_name}/{subset}")
    #     os.makedirs(EN_synthesis_result_dir, exist_ok=True)
    #
    #     synthesis(config_file, checkpoint_file, EN_wav_list, EN_text_list, EN_unit_seq_list,
    #           EN_synthesis_result_dir, mode, unit_type, print_verbose_info=False)


    # CH_synthesis_result_dir = Path(
    #     f"/home/ma-user/work/daxintan/temp_speech_reconstruction/CH_streaming/{CH_dataset}_{CH_subset}_{unit_type}")
    # synthesis(CH_model_config_file, CH_model_checkpoint_file, CH_wav_list, CH_text_list, CH_unit_seq_list,
    #           CH_synthesis_result_dir, mode, unit_type, print_verbose_info=False, streaming=True)
