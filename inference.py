import os
import json
import math
import argparse
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import json


class DummyLoader(TextAudioSpeakerLoader):
    def __init__(self, text_file, vc_file, hparams, text_sid=False):
        self.text_file = text_file
        self.vc_file = vc_file
        self.text_sid = text_sid
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.merge_file()

    def merge_file(self):
        texts = [li.strip().split('|') for li in open(self.text_file)]
        pairs = [json.loads(li.strip()) for li in open(self.vc_file)]
        assert len(pairs) == len(texts), "Pairs"

        n2text = {}
        for li in texts:
            if self.text_sid:
                n, _, text = li
            else:
                n, text = li
            n2text[n] = text

        self.audiopaths_and_text = []
        for i, pair in enumerate(pairs):
            src = pair['src']
            text = n2text[src]
            for j, tar in enumerate(pair['tar']):
                item = {
                    "src": src,
                    "text": text,
                    "tar": tar,
                    "names": [
                        f"{i}_src.wav" if j == 0 else None,
                        f"{i}_vc{j}.wav",
                        f"{i}_tar{j}.wav"]
                }
                self.audiopaths_and_text.append(item)

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        """
            src, src_text, src_wav, tar, tar_spec, tar_wav, (src_name, vc_name, tar_name)
        """
        item = audiopath_and_text
        src, tar = item["src"], item["tar"]
        names = item["names"]
        src_text = self.get_text(item["text"])

        src_wav = None
        if src is not None and os.path.exists(src) and names[0] is not None:
            _, src_wav = self.get_audio(src)

        tar_spec, tar_wav = self.get_audio(tar)
        return src, src_text, src_wav, tar, tar_spec, tar_wav, names

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def main(args):
    hps = utils.get_hparams_from_file(args.config)

    noise_config = {
        "noise_scale": args.noise_scale,
        "noise_scale_w": args.noise_scale_w,
        "length_scale": args.length_scale,
        "max_len": args.max_len
    }
    print("noise_config: ", noise_config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,  # multi-speaker
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.model, net_g, None)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    n_speakers = hps.data.n_speakers
    use_ref_enc = hps.model.get("use_ref_enc", False)

    vc = args.vc_file is not None
    if vc:
        eval_dataset = DummyLoader(args.input, args.vc_file, hps.data, text_sid=n_speakers > 0)

        for i, aux in enumerate(tqdm(eval_dataset)):
            src, src_text, src_wav, tar, tar_spec, tar_wav, (src_name, vc_name, tar_name) = aux
            x, x_lengths = src_text.unsqueeze(0).cuda(), torch.LongTensor([src_text.shape[0]]).cuda()
            spec, spec_lengths = tar_spec.unsqueeze(0).cuda(), torch.LongTensor([tar_spec.shape[1]]).cuda()
            speakers = None
            if use_ref_enc:
                y_hat, attn, mask, *_ = net_g.infer(x, x_lengths, speakers, spec=spec, spec_lengths=spec_lengths,
                                                    **noise_config)
            else:
                y_hat, attn, mask, *_ = net_g.infer(x, x_lengths, speakers, **noise_config)

            # write waveform
            y_hat = y_hat.data.cpu().float().numpy().reshape(-1)
            out_path = os.path.join(args.output, vc_name)
            write(out_path, hps.data.sampling_rate, y_hat)

            wav = tar_wav.numpy().reshape(-1)
            out_path = os.path.join(args.output, tar_name)
            write(out_path, hps.data.sampling_rate, wav)

            if src_wav is not None:
                src_wav = src_wav.numpy().reshape(-1)
                out_path = os.path.join(args.output, src_name)
                write(out_path, hps.data.sampling_rate, src_wav)
    else:
        input_file = args.input
        if n_speakers > 0:
            eval_dataset = TextAudioSpeakerLoader(input_file, hps.data, is_training=False)
        else:
            eval_dataset = TextAudioLoader(input_file, hps.data, is_training=False)

        for idx, aux in enumerate(tqdm(eval_dataset)):
            if n_speakers > 0:
                text, spec, wav, sid = aux
            else:
                text, spec, wav = aux

            x, x_lengths = text.unsqueeze(0).cuda(), torch.LongTensor([text.shape[0]]).cuda()
            spec, spec_lengths = spec.unsqueeze(0).cuda(), torch.LongTensor([spec.shape[1]]).cuda()
            # y, y_lengths = wav.unsqueeze(0).cuda(), torch.LongTensor([wav.shape[1]]).cuda()
            if n_speakers > 0:
                speakers = torch.LongTensor([sid]).cuda()
            else:
                speakers = None

            if use_ref_enc:
                y_hat, attn, mask, *_ = net_g.infer(x, x_lengths, speakers, spec=spec, spec_lengths=spec_lengths,
                                                    **noise_config)
            else:
                y_hat, attn, mask, *_ = net_g.infer(x, x_lengths, speakers, **noise_config)

            y_hat = y_hat.data.cpu().float().numpy().reshape(-1)
            out_path = os.path.join(args.output, f"{idx}_pred.wav")
            write(out_path, hps.data.sampling_rate, y_hat)

            wav = wav.numpy().reshape(-1)
            out_path = os.path.join(args.output, f"{idx}_gt.wav")
            write(out_path, hps.data.sampling_rate, wav)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    parser.add_argument('-o', '--output', type=str, required=True, help='output dir')
    parser.add_argument('-i', '--input', type=str, required=True, help='input file')
    parser.add_argument('-v', '--vc-file', type=str, default=None)
    parser.add_argument('--noise_scale', type=float, default=0.667)
    parser.add_argument('--noise_scale_w', type=float, default=0.8)
    parser.add_argument('--length_scale', type=float, default=1)
    parser.add_argument('--max_len', type=int, default=2000)
    args = parser.parse_args()
    main(args)
