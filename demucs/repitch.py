# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import random
import subprocess as sp
import tempfile

import numpy as np
import torch
from scipy.io import wavfile


def i16_pcm(wav):
    if wav.dtype == np.int16:
        return wav
    return (wav * 2**15).clamp_(-2**15, 2**15 - 1).short()


def f32_pcm(wav):
    if wav.dtype == np.float:
        return wav
    return wav.float() / 2**15


class RepitchedWrapper:
    """
    Wrap a dataset to apply online change of pitch / tempo.
    """
    def __init__(self, dataset, proba=0.2, max_pitch=2, max_tempo=12, tempo_std=5, vocals=[3]):
        self.dataset = dataset
        self.proba = proba
        self.max_pitch = max_pitch
        self.max_tempo = max_tempo
        self.tempo_std = tempo_std
        self.vocals = vocals

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        streams = self.dataset[index]
        in_length = streams.shape[-1]
        out_length = int((1 - 0.01 * self.max_tempo) * in_length)

        if random.random() < self.proba:
            delta_pitch = random.randint(-self.max_pitch, self.max_pitch)
            delta_tempo = random.gauss(0, self.tempo_std)
            delta_tempo = min(max(-self.max_tempo, delta_tempo), self.max_tempo)
            outs = []
            for idx, stream in enumerate(streams):
                stream = repitch(
                    stream,
                    delta_pitch,
                    delta_tempo,
                    voice=idx in self.vocals)
                outs.append(stream[:, :out_length])
            streams = torch.stack(outs)
        else:
            streams = streams[..., :out_length]
        return streams


def repitch(wav, pitch, tempo, voice=False, quick=False, samplerate=44100):
    """
    tempo is a relative delta in percentage, so tempo=10 means tempo at 110%!
    pitch is in semi tones.
    Requires `soundstretch` to be installed, see
    https://www.surina.net/soundtouch/soundstretch.html
    """
    outfile = tempfile.NamedTemporaryFile(suffix=".wav")
    in_ = io.BytesIO()
    wavfile.write(in_, samplerate, i16_pcm(wav).t().numpy())
    command = [
        "soundstretch",
        "stdin",
        outfile.name,
        f"-pitch={pitch}",
        f"-tempo={tempo:.6f}",
    ]
    if quick:
        command += ["-quick"]
    if voice:
        command += ["-speech"]
    try:
        sp.run(command, capture_output=True, input=in_.getvalue(), check=True)
    except sp.CalledProcessError as error:
        raise RuntimeError(f"Could not change bpm because {error.stderr.decode('utf-8')}")
    sr, wav = wavfile.read(outfile.name)
    wav = wav.copy()
    wav = f32_pcm(torch.from_numpy(wav).t())
    assert sr == samplerate
    return wav
