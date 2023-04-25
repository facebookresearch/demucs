# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API calls
"""

import random
import subprocess
import warnings

import torch as th
import torchaudio as ta

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Hashable, Any

from .apply import BagOfModels, tensor_chunk, TensorChunk
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model
from .separate import get_parser
from .utils import center_trim, DummyPoolExecutor


class LoadAudioError(Exception):
    pass


def _replace_dict(_dict: dict, *subs: tuple[Hashable, Any]):
    for key, value in subs:
        _dict[key] = value
    return _dict


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
                    "When trying to load using {}, got the following error: {}".format(
                        backend, error
                    )
                    for backend, error in errors.items()
                )
            )
        return wav

    def load_audios(self, *tracks, audio_channels=None, samplerate=None, ignore_errors=True):
        for track in tracks:
            if ignore_errors:
                try:
                    yield track, self._load_audio(track, audio_channels, samplerate)
                except Exception:
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

    def load_audios_from_cmdline(self):
        self.load_audios_to_model(*self._opts.tracks)

    def clear_filelist(self):
        self._wav = []
        self._file = []

    def add_track(self, filename, wav):
        self._file.append(filename)
        self._wav.append(wav)

    def separate_track(
        self,
        wav,
        model=None,
        segment=0,
        shifts=None,
        split=None,
        overlap=None,
        transition_power=1.0,
        device=None,
        num_workers=None,
        pool=None,
        callback: Callable[[dict], None] = None,
        callback_arg: dict = None,
    ):
        if model is None:
            if self._model is None:
                raise RuntimeError("Load a model first! ")
            model = self._model
        if not segment:
            segment = self._opts.segment
        if shifts is None:
            shifts = self._opts.shifts
        if split is None:
            split = self._opts.split
        if overlap is None:
            overlap = self._opts.overlap
        if device is None:
            device = self.device
        if num_workers is None:
            num_workers = self._opts.jobs
        if pool is None:
            if num_workers > 0 and device.type == "cpu":
                pool = ThreadPoolExecutor(num_workers)
            else:
                pool = DummyPoolExecutor()
        if callback_arg is None:
            callback_arg = {}
        callback_arg = _replace_dict(
            callback_arg, *{"model_idx_in_bag": 0, "shift_idx": 0, "segment_offset": 0}.items()
        )
        kwargs = {
            "shifts": shifts,
            "split": split,
            "overlap": overlap,
            "transition_power": transition_power,
            "device": device,
            "pool": pool,
            "segment": segment,
        }
        if isinstance(model, BagOfModels):
            estimates = 0
            totals = [0] * len(model.sources)
            callback_arg["models"] = len(model.models)
            for sub_model, weight in zip(model.models, model.weights):
                original_model_device = next(iter(sub_model.parameters())).device
                sub_model.to(device)

                out = self.separate_track(
                    sub_model,
                    wav,
                    **kwargs,
                    callback_arg=callback_arg,
                    callback=(
                        lambda d, i=callback_arg["model_idx_in_bag"]: callback(
                            _replace_dict(d, ("model_idx_in_bag", i))
                        )
                    )
                    if callable(callback)
                    else None,
                )
                sub_model.to(original_model_device)
                for k, inst_weight in enumerate(weight):
                    out[:, k, :, :] *= inst_weight
                    totals[k] += inst_weight
                estimates += out
                del out
                callback_arg["model_idx_in_bag"] += 1

            for k in range(estimates.shape[1]):
                estimates[:, k, :, :] /= totals[k]
            return estimates

        callback_arg["models"] = 1
        model.to(device)
        model.eval()
        assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
        batch, channels, length = wav.shape
        if shifts:
            kwargs["shifts"] = 0
            max_shift = int(0.5 * model.samplerate)
            wav = tensor_chunk(wav)
            padded_mix = wav.padded(length + 2 * max_shift)
            out = 0
            for shift_idx in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
                shifted_out = self.separate_track(
                    model,
                    shifted,
                    **kwargs,
                    callback_arg=callback_arg,
                    callback=(lambda d, i=shift_idx: callback(_replace_dict(d, ("shift_idx", i))))
                    if callable(callback)
                    else None,
                )
                out += shifted_out[..., max_shift - offset :]
            out /= shifts
            return out
        elif split:
            kwargs["split"] = False
            out = th.zeros(batch, len(model.sources), channels, length, device=wav.device)
            sum_weight = th.zeros(length, device=wav.device)
            if segment is None:
                segment = model.segment
            segment = int(model.samplerate * segment)
            stride = int((1 - overlap) * segment)
            offsets = range(0, length, stride)
            # We start from a triangle shaped weight, with maximal weight in the middle
            # of the segment. Then we normalize and take to the power `transition_power`.
            # Large values of transition power will lead to sharper transitions.
            weight = th.cat(
                [
                    th.arange(1, segment // 2 + 1, device=device),
                    th.arange(segment - segment // 2, 0, -1, device=device),
                ]
            )
            assert len(weight) == segment
            # If the overlap < 50%, this will translate to linear transition when
            # transition_power is 1.
            weight = (weight / weight.max()) ** transition_power
            futures = []
            for offset in offsets:
                chunk = TensorChunk(wav, offset, segment)
                future = pool.submit(
                    self.separate_track,
                    model,
                    chunk,
                    **kwargs,
                    callback_arg=callback_arg,
                    callback=(lambda d, i=offset: callback(_replace_dict(d, ("segment_offset", i))))
                    if callable(callback)
                    else None,
                )
                futures.append((future, offset))
                offset += segment
            for future, offset in futures:
                chunk_out = future.result()
                chunk_length = chunk_out.shape[-1]
                out[..., offset : offset + segment] += (weight[:chunk_length] * chunk_out).to(
                    wav.device
                )
                sum_weight[offset : offset + segment] += weight[:chunk_length].to(wav.device)
            assert sum_weight.min() > 0
            out /= sum_weight
            return out
        else:
            if hasattr(model, "valid_length"):
                valid_length = model.valid_length(length)
            else:
                valid_length = length
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()
            wav = tensor_chunk(wav[None])
            padded_mix = wav.padded(valid_length).to(device)
            if callable(callback):
                callback(_replace_dict(callback_arg, ("state", "start")))
            with th.no_grad():
                out = model(padded_mix)
            if callable(callback):
                callback(_replace_dict(callback_arg, ("state", "end")))
            return center_trim(out, length) * ref.std() + ref.mean()

    def separate_loaded_audio(
        self,
        segment=0,
        split=None,
        shifts=None,
        overlap=None,
        transition_power=1.0,
        device=None,
        num_workers=None,
        callback: Callable[[dict], None] = None,
        callback_arg: dict = None,
    ):
        if len(self._file) != len(self._wav):
            raise RuntimeError("File list and waves not matched. Please `clear_filelist` first. ")
        kwargs = {
            "model": self._model,
            "shifts": shifts,
            "split": split,
            "overlap": overlap,
            "transition_power": transition_power,
            "device": device,
            "segment": segment,
            "num_workers": num_workers,
        }
        self._out = []
        for file, wav in zip(self._file, self._wav):
            out = self.separate_track(
                wav,
                **kwargs,
                callback_arg=_replace_dict(callback_arg, ("file", file)),
                callback=callback,
            )
            self._out.append(file, dict(zip(out, self._model.sources)))
            yield file, dict(zip(out, self._model.sources))


if __name__ == "__main__":
    # Test API functions
    # two-stem not supported

    import pathlib
    import sys

    separator = Separator(sys.argv[1:])
    separator.load_model()
    separator.load_audios_from_cmdline()
    args = separator._opts
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    for file, sources in separator.separate_loaded_audio(callback=print):
        if args.mp3:
            ext = "mp3"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator._samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        for stem, source in sources.items():
            stem = out / args.filename.format(
                track=pathlib.Path(file).name.rsplit(".", 1)[0],
                trackext=pathlib.Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)
