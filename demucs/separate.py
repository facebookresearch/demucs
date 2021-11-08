# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

from dora.log import fatal
import torch as th
import torchaudio as ta

from .apply import apply_model, BagOfModels
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import get_model_from_args, add_model_flags, ModelLoadingError


def load_track(track, device, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels).to(device)
    except FileNotFoundError:
        errors['ffmpeg'] = 'Ffmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = wav.to(device)
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav


def main():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    add_model_flags(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=1,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--overlap",
                        default=0.25,
                        type=float,
                        help="Overlap between the splits.")
    parser.add_argument("--no-split",
                        action="store_false",
                        dest="split",
                        default=True,
                        help="Doesn't split audio in chunks. This can use large amounts of memory.")
    parser.add_argument("--mp3", action="store_true",
                        help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")

    args = parser.parse_args()

    try:
        model = get_model_from_args(args)
    except ModelLoadingError as error:
        fatal(error.args[0])

    if isinstance(model, BagOfModels):
        print(f"Selected model is a bag of {len(model.models)} models. "
              "You will see that many progress bars per track.")
    model.to(args.device)
    model.eval()

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        print(f"Separating track {track}")
        wav = load_track(track, args.device, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav[None], shifts=args.shifts, split=args.split,
                              overlap=args.overlap, progress=True)[0]
        sources = sources * ref.std() + ref.mean()

        track_folder = out / track.name.rsplit(".", 1)[0]
        track_folder.mkdir(exist_ok=True)
        for source, name in zip(sources, model.sources):
            source = source.cpu()
            stem = str(track_folder / name)
            if args.mp3:
                stem += ".mp3"
            else:
                stem += ".wav"
            save_audio(source, stem, model.samplerate)


if __name__ == "__main__":
    main()
