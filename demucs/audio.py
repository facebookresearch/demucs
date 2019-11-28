# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
import subprocess as sp
import tempfile
from pathlib import Path

import numpy as np
import torch


def _read_info(path):
    process = sp.run(
        ['ffprobe', str(path), '-print_format', 'json', '-show_format', '-show_streams'],
        capture_output=True,
        check=True)
    return json.loads(process.stdout.decode('utf-8'))


class AudioFile:
    """
    Allows to read audio from any format supported by ffmpeg, as well as resampling or
    converting to mono on the fly. See :method:`read` for more details.
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self._info = None

    def __repr__(self):
        features = [("path", self.path)]
        features.append(("samplerate", self.samplerate()))
        features.append(("channels", self.channels()))
        features.append(("streams", len(self)))
        features_str = ", ".join(f"{name}={value}" for name, value in features)
        return f"AudioFile({features_str})"

    @property
    def info(self):
        if self._info is None:
            self._info = _read_info(self.path)
        return self._info

    @property
    def duration(self):
        return float(self.info['format']['duration'])

    @property
    def _audio_streams(self):
        return [
            index for index, stream in enumerate(self.info["streams"])
            if stream["codec_type"] == "audio"
        ]

    def __len__(self):
        return len(self._audio_streams)

    def channels(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['channels'])

    def samplerate(self, stream=0):
        return int(self.info['streams'][self._audio_streams[stream]]['sample_rate'])

    def read(self,
             seek_time=None,
             duration=None,
             streams=slice(None),
             samplerate=None,
             channels=None):
        """
        Slightly more efficient implementation than stempeg,
        in particular, this will extract all stems at once
        rather than having to loop over one file multiple times
        for each stream.

        Args:
            seek_time (float):  seek time in seconds or None if no seeking is needed.
            duration (float): duration in seconds to extract or None to extract until the end.
            streams (slice, int or list): streams to extract, can be a single int, a list or
                a slice. If it is a slice or list, the output will be of size [S, C, T]
                with S the number of streams, C the number of channels and T the number of samples.
                If it is an int, the output will be [C, T].
            samplerate (int): if provided, will resample on the fly. If None, no resampling will
                be done. Original sampling rate can be obtained with :method:`samplerate`.
            channels (int): if 1, will convert to mono. We do not rely on ffmpeg for that
                as ffmpeg automatically scale by +3dB to conserve volume when playing on speakers.
                See https://sound.stackexchange.com/a/42710.
                Our definition of mono is simply the average of the two channels. Any other
                value will be ignored.


        """
        streams = np.array(range(len(self)))[streams]
        single = not isinstance(streams, np.ndarray)
        if single:
            streams = [streams]

        if duration is None:
            target_size = None
            query_duration = None
        else:
            target_size = int((samplerate or self.samplerate()) * duration)
            query_duration = float((target_size + 1) / (samplerate or self.samplerate()))

        shm = Path("/dev/shm")
        if shm.exists():
            temp_folder = shm / os.environ['USER']
            temp_folder.mkdir(exist_ok=True)
        else:
            temp_folder = None
        files = [tempfile.NamedTemporaryFile(dir=temp_folder) for _ in streams]
        command = ['ffmpeg', '-y']
        command += ['-loglevel', 'panic']
        if seek_time:
            command += ['-ss', str(seek_time)]
        command += ['-i', str(self.path)]
        for stream, file in zip(streams, files):
            command += ['-map', f'0:{self._audio_streams[stream]}']
            if query_duration is not None:
                command += ['-t', str(query_duration)]
            command += ['-threads', '1']
            command += ['-f', 'f32le']
            if samplerate is not None:
                command += ['-ar', str(samplerate)]
            command += [file.name]

        sp.run(command, check=True)
        wavs = []
        for file in files:
            wav = np.fromfile(file.name, dtype=np.float32)
            wav = torch.from_numpy(wav)
            wav = wav.view(-1, self.channels()).t()
            if channels == 1:
                # We do mono convertion here as ffmpeg mess up the volume of mono output
                # otherwise. See https://sound.stackexchange.com/a/42710.
                wav = wav.mean(dim=0, keepdim=True)
            if target_size is not None:
                wav = wav[..., :target_size]
            wavs.append(wav)
        wav = torch.stack(wavs, dim=0)
        if single:
            wav = wav[0]
        return wav
