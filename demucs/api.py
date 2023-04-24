# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API calls
"""

import subprocess
import warnings

import torchaudio as ta

from .apply import BagOfModels
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model
from .separate import get_parser


class LoadAudioError(Exception):
    pass


class Separator:
    def __init__(self, cmd_line, **kw):
        self._opts = get_parser().parse_args(cmd_line if cmd_line else [""])
        if not cmd_line:
            self._opts.tracks = []
        for key, value in kw:
            setattr(self._opts, key, value)
        self._model = None
        self._audio_channels = 2
        self._samplerate = 44100
        self.segment = None
        self._wav = []
        self._file = []
        self.device = self._opts.device

    def load_model(self, model=None, repo=None):
        if self._model is not None:
            raise RuntimeError("Method `load_model` can only be called once. ")
        if model is not None:
            self._opts.name = model
        if repo is not None:
            self._opts.repo = repo
        self._model = get_model(name=self._opts.name, repo=self._opts.repo)
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate
        return self._model

    def _load_audio(self, track, audio_channels=None, samplerate=None):
        errors = {}
        wav = None
        if audio_channels is None:
            audio_channels = self._audio_channels
        if samplerate is None:
            samplerate = self._samplerate

        try:
            wav = AudioFile(track).read(streams=0, samplerate=samplerate, channels=audio_channels)
        except FileNotFoundError:
            errors["ffmpeg"] = "FFmpeg is not installed."
        except subprocess.CalledProcessError:
            errors["ffmpeg"] = "FFmpeg could not read the file."

        if wav is None:
            try:
                wav, sr = ta.load(str(track))
            except RuntimeError as err:
                errors["torchaudio"] = err.args[0]
            else:
                wav = convert_audio(wav, sr, samplerate, audio_channels)

        if wav is None:
            raise LoadAudioError(
                "\n".join(
                    "When trying to load using {}, got the following error: {}".format(backend, error)
                    for backend, error in errors.items()
                )
            )
        return wav

    def load_audios(self, *tracks, audio_channels=None, samplerate=None, ignore_errors=True):
        for track in tracks:
            if ignore_errors:
                try:
                    yield track, self._load_audio(track, audio_channels, samplerate)
                except:
                    yield track, None
            else:
                yield track, self._load_audio(track, audio_channels, samplerate)

    def load_audios_to_model(self, *tracks):
        if self._model is None:
            raise RuntimeError("Please load model first! ")
        for track, wav in self.load_audios(*tracks):
            if wav is None:
                warnings.warn("Audio read failed and will not be loaded to model: " + track)
                continue
            self._file.append(track)
            self._wav.append(wav)

    def clear_filelist(self):
        self._wav = []
        self._file = []

    def add_track(self, filename, wav):
        self._file.append(filename)
        self._wav.append(wav)

