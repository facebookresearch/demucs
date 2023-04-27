# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API methods for demucs

Classes
-------
`demucs.api.Separator`: The base separator class

Functions
---------
`demucs.api.save_audio`: Save an audio

Examples
--------
See the end of this module (if __name__ == "__main__")
"""

import random
import subprocess
import warnings

import torch as th
import torchaudio as ta

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable, List, Any, Hashable, Tuple

from .apply import BagOfModels, tensor_chunk, TensorChunk
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model
from .repo import AnyModel
from .separate import get_parser
from .utils import center_trim, DummyPoolExecutor


class LoadAudioError(Exception):
    pass


class LoadModelError(Exception):
    pass


def _replace_dict(_dict: Optional[dict], *subs: Tuple[Hashable, Any]) -> dict:
    if _dict is None:
        _dict = {}
    for key, value in subs:
        _dict[key] = value
    return _dict


class Separator:
    def __init__(self, cmd_line: Optional[List[str]] = None, **kw):
        """
        `class Separator`
        =================

        Parameters
        ----------
        cmd_line (optional): Parsed command line, use `sys.argv[1:]` to use runtime command line. \
            Supported commands are same as `demucs.separate`. Please remember that not all the \
            options are supported.
        kw: Arguments to be added or replaced in the command line. To get a list of supported \
            keys, you can run `print(dir(demucs.separate.get_parser().parse_args([""])))`.
        """
        self._opts = get_parser().parse_args(cmd_line if cmd_line else [""])
        if not cmd_line:
            self._opts.tracks = []
        for key, value in kw.items():
            setattr(self._opts, key, value)
        self._model = None
        self._audio_channels = 2
        self._samplerate = 44100
        self.segment = None
        self._wav: List[th.Tensor] = []
        self._file: List[str] = []
        self.device = self._opts.device

    def load_model(self, model: Optional[str] = None, repo: Optional[str] = None):
        """
        Load a model to the class and return the model. This could only be called once.

        To manually add a loaded model to the class, simply assign the `Separator._model` variable.

        Parameters
        ----------
        model: If not specified, will use the model specified in the command line.
        repo: If not specified, will use the model specified in the command line.

        Returns
        -------
        Model (Demucs | HDemucs | HTDemucs | BagOfModels)
        """
        if self._model is not None:
            raise RuntimeError("Method `load_model` can only be called once. ")
        if model is not None:
            self._opts.name = model
        if repo is not None:
            self._opts.repo = repo
        self._model = get_model(name=self._opts.name, repo=self._opts.repo)
        if self._model is None:
            raise LoadModelError("Failed to load model")
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate
        return self._model

    def _load_audio(
        self,
        track: Path,
        audio_channels: Optional[int] = None,
        samplerate: Optional[int] = None,
    ):
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

    def load_audios(
        self,
        *tracks: Path,
        audio_channels: Optional[int] = None,
        samplerate: Optional[int] = None,
        ignore_errors=True,
    ):
        """
        Load several audios and return the iterator of the audios. This function returns an
        iterator, so the audio will not be decoded and read into memory until the iterator reached
        the spcefic item.

        If you want to just read one audio, you can use `next(iter(Separator.load_audios(path)))`

        If you want to load several audio at once, please use `list(Separator.load_audios(path))`

        Parameters
        ----------
        tracks: Pathlike objects, containing the path of the audio to be loaded.
        audio_channels: The targeted audio channels. If not specified, will use the value of the \
            loaded model. If no model is loaded, 2 is the default value.
        samplerate: The targeted audio channels. If not specified, will use the value of the \
            loaded model. If no model is loaded, 44100 is the default value.
        ignore_errors: If true, any exception encountered will be ignored and the audio failed to \
            be loaded will become `None`.

        Returns
        -------
        A generator (iterator) of tuple[filename, wave]. If `ignore_errors` is True (default), the
        wave of that file will be `None`.
        """
        for track in tracks:
            if ignore_errors:
                try:
                    yield track, self._load_audio(track, audio_channels, samplerate)
                except Exception:
                    yield track, None
            else:
                yield track, self._load_audio(track, audio_channels, samplerate)

    def load_audios_to_model(self, *tracks: Path):
        """
        Load several audios to the Separator that can be used in `Separator.separate_loaded_audio`.

        Parameters
        ----------
        tracks: Pathlike objects, containing the path of the audio to be loaded.

        Returns
        -------
        None

        Notes
        -----
        When an error encountered, this function will only warn and continue loading other audios.
        The audio failed to load will not be added into the Separator. To get a list of audios
        failed to be loaded, you can use the following codes:
        ```python
        import warnings
        with warnings.catch_warnings(record=True) as w:
            Separator.load_audios_to_model(track)
            failures = list(i.message.separate('"')[1] for i in w)
        ```
        """
        if self._model is None:
            raise RuntimeError("Please load model first! ")
        for track, wav in self.load_audios(*tracks):
            if wav is None:
                warnings.warn(f'"{track}" read failed and will not be loaded to model. ')
                continue
            self._file.append(track)
            self._wav.append(wav)

    def load_audios_from_cmdline(self):
        """
        Load audios specied in the command line to the Separator that can be used in
        `Separator.separate_loaded_audio`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        When an error encountered, this function will only warn and continue loading other audios.
        The audio failed to load will not be added into the Separator. To get a list of audios
        failed to be loaded, you can use the following codes:
        ```python
        import warnings
        with warnings.catch_warnings(record=True) as w:
            Separator.load_audios_from_cmdline()
            failures = list(i.message.separate('"')[1] for i in w)
        ```
        """
        self.load_audios_to_model(*self._opts.tracks)

    def clear_filelist(self):
        """
        Remove all the loaded audios in the Separator.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._wav = []
        self._file = []

    def add_track(self, filename: str, wav: th.FloatTensor):
        """
        Add a loaded track into the separator.

        Parameters
        ----------
        filename: A string for you to remember what the each audio is. (You can call it \
            "identifier")
        wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, \
            while the second is the waveform of each channel. \
            e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.

        Returns
        -------
        None

        Notes
        -----
        Use this function with cautiousness. This function does not provide data verifying.
        """
        self._file.append(filename)
        self._wav.append(wav)

    def _separate_track(
        self,
        wav,
        model: Optional[AnyModel] = None,
        segment: Optional[float] = 0.0,
        shifts: Optional[int] = None,
        split: Optional[bool] = None,
        overlap: Optional[float] = None,
        transition_power=1.0,
        device=None,
        num_workers=None,
        pool=None,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ) -> th.Tensor:
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
            estimates = th.Tensor()
            totals = [0.0] * len(model.sources)
            callback_arg["models"] = len(model.models)
            kwargs["callback"] = (
                (
                    lambda d, i=callback_arg["model_idx_in_bag"]: callback(
                        _replace_dict(d, ("model_idx_in_bag", i))
                    )
                )
                if callable(callback)
                else None
            )
            for sub_model, weight in zip(model.models, model.weights):
                original_model_device = next(iter(sub_model.parameters())).device
                sub_model.to(device)

                out = self._separate_track(
                    wav,
                    model=sub_model,
                    **kwargs,
                    callback_arg=callback_arg,
                )
                sub_model.to(original_model_device)
                for k, inst_weight in enumerate(weight):
                    out[:, k, :, :] *= inst_weight
                    totals[k] += inst_weight
                if not len(estimates):
                    estimates = out
                else:
                    estimates = estimates + out
                del out
                callback_arg["model_idx_in_bag"] += 1

            for k in range(estimates.shape[1]):
                estimates[:, k, :, :] /= totals[k]
            return estimates

        if "models" not in callback_arg:
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
            out = th.Tensor()
            for shift_idx in range(shifts):
                offset = random.randint(0, max_shift)
                shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
                kwargs["callback"] = (
                    (lambda d, i=shift_idx: callback(_replace_dict(d, ("shift_idx", i))))
                    if callable(callback)
                    else None
                )
                shifted_out = self._separate_track(
                    shifted,
                    model=model,
                    **kwargs,
                    callback_arg=callback_arg,
                )
                if not len(out):
                    out = shifted_out[..., max_shift - offset:]
                else:
                    out = out + shifted_out[..., max_shift - offset:]
            out /= shifts
            model.cpu()
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
            weight_seg = th.cat(
                [
                    th.arange(1, segment // 2 + 1, device=device),
                    th.arange(segment - segment // 2, 0, -1, device=device),
                ]
            )
            assert len(weight_seg) == segment
            # If the overlap < 50%, this will translate to linear transition when
            # transition_power is 1.
            weight_seg = (weight_seg / weight_seg.max()) ** transition_power
            futures = []
            for offset in offsets:
                chunk = TensorChunk(wav, offset, segment)
                future = pool.submit(
                    self._separate_track,
                    chunk,
                    model=model,
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
                out[..., offset: offset + segment] += (weight_seg[:chunk_length] * chunk_out).to(
                    wav.device
                )
                sum_weight[offset: offset + segment] += weight_seg[:chunk_length].to(wav.device)
            assert sum_weight.min() > 0
            out /= sum_weight
            model.cpu()
            return out
        else:
            if hasattr(model, "valid_length") and callable(model.valid_length):
                valid_length = model.valid_length(length)
            else:
                valid_length = length
            wav = tensor_chunk(wav)
            padded_mix = wav.padded(valid_length).to(device)
            if callable(callback):
                callback(_replace_dict(callback_arg, ("state", "start")))
            with th.no_grad():
                out = model(padded_mix)
            if callable(callback):
                callback(_replace_dict(callback_arg, ("state", "end")))
            model.cpu()
            return center_trim(out, length)

    def separate_audio(
        self,
        wav,
        model: Optional[AnyModel] = None,
        segment: Optional[float] = 0.0,
        shifts: Optional[int] = None,
        split: Optional[bool] = None,
        overlap: Optional[float] = None,
        transition_power=1.0,
        device=None,
        num_workers=None,
        pool=None,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """
        Separate an audio.

        Parameters
        ----------
        wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, \
            while the second is the waveform of each channel. \
            e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.
        model: Model to be used. If not specified, will use the model loaded to the Separator.
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option.
        num_workers: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.

        Returns
        -------
        Separated stems.

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.

        Notes
        -----
        Use this function with cautiousness. This function does not provide data verifying.
        """
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        return (
            self._separate_track(
                wav[None],
                model=model,
                segment=segment,
                shifts=shifts,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                device=device,
                num_workers=num_workers,
                pool=pool,
                callback=callback,
                callback_arg=_replace_dict(callback_arg, ("audio_length", wav.shape[1])),
            )
            * ref.std()
            + ref.mean()
        )

    def separate_loaded_audio(
        self,
        segment: Optional[float] = 0.0,
        shifts: Optional[int] = None,
        split: Optional[bool] = None,
        overlap: Optional[float] = None,
        transition_power=1.0,
        device=None,
        num_workers=None,
        callback: Optional[Callable[[dict], None]] = None,
        callback_arg: Optional[dict] = None,
    ):
        """
        Separate the audio loaded into the Separator.

        Parameters
        ----------
        segment: Length (in seconds) of each segment (only available if `split` is `True`). If \
            not specified, will use the command line option.
        shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and \
            apply the oppositve shift to the output. This is repeated `shifts` time and all \
            predictions are averaged. This effectively makes the model time equivariant and \
            improves SDR by up to 0.2 points. If not specified, will use the command line option.
        split: If True, the input will be broken down into small chunks (length set by `segment`) \
            and predictions will be performed individually on each and concatenated. Useful for \
            model with large memory footprint like Tasnet. If not specified, will use the command \
            line option.
        overlap: The overlap between the splits. If not specified, will use the command line \
            option.
        device (torch.device, str, or None): If provided, device on which to execute the \
            computation, otherwise `wav.device` is assumed. When `device` is different from \
            `wav.device`, only local computations will be on `device`, while the entire tracks \
            will be stored on `wav.device`. If not specified, will use the command line option.
        num_workers: Number of jobs. This can increase memory usage but will be much faster when \
            multiple cores are available. If not specified, will use the command line option.
        callback: A function will be called when the separation of a chunk starts or finished. \
            The argument passed to the function will be a dict. For more information, please see \
            the Callback section.
        callback_arg: A dict containing private parameters to be passed to callback function. For \
            more information, please see the Callback section.

        Returns
        -------
        A generator (iterator) of tuple[filename, dict[stem_name, waveform]].

        Callback
        --------
        The function will be called with only one positional parameter whose type is `dict`. The
        `callback_arg` will be combined with information of current separation progress. The
        progress information will override the values in `callback_arg` if same key has been used.

        Progress information contains several keys (These keys will always exist):
        - `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
        - `shift_idx`: The index of shifts. Starts from 0.
        - `segment_offset`: The offset of current segment. If the number is 441000, it doesn't
            mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
        - `state`: Could be `"start"` or `"end"`.
        - `audio_length`: Length of the audio (in "frame" of the tensor).
        - `models`: Count of submodels in the model.
        - `file`: File name (or what you may call "identifier") of the waveform that is being
            separated

        Notes
        -----
        The returns of the function is an iterator, so the separation process will not start until
        the iterator reaches the specific item.

        To separate the first audio in the loaded list, use
        `next(iter(Separator.separate_loaded_audio()))`.

        To separate all the audio at once (which may consume lots of memory), use
        `list(Separator.separate_loaded_audio())`.

        The function will not remove the separated tracks from the Separator. So run
        `Separator.clear_filelist()` if you like to remove them.

        To get the separated tracks, just retrieve `Separator._out` variable. It will be cleared
        each time you run this function.
        """
        if len(self._file) != len(self._wav):
            raise RuntimeError("File list and waves not matched. Please `clear_filelist` first. ")
        if self._model is None:
            raise RuntimeError("Load a model first! ")
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
            out = self.separate_audio(
                wav,
                **kwargs,
                callback_arg=_replace_dict(callback_arg, ("file", file)),
                callback=callback,
            )
            self._out.append((file, dict(zip(self._model.sources, out[0]))))
            yield file, dict(zip(self._model.sources, out[0]))


if __name__ == "__main__":
    # Test API functions
    # two-stem not supported

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
                track=Path(file).name.rsplit(".", 1)[0],
                trackext=Path(file).name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(source, str(stem), **kwargs)
