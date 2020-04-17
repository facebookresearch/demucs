# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import sys
from pathlib import Path

import requests
import torch as th
import tqdm
from scipy.io import wavfile

from .audio import AudioFile
from .utils import apply_model, load_model

BASE_URL = "https://dl.fbaipublicfiles.com/demucs/v2.0/"
PRETRAINED_MODELS = {
    'demucs.th': 'f6c4148ba0dc92242d82d7b3f2af55c77bd7cb4ff1a0a3028a523986f36a3cfd',
    'demucs.th.gz': 'e70767bfc9ce62c26c200477ea29a20290c708b210977e3ef2c75ace68ea4be1',
    'demucs_extra.th': '3331bcc5d09ba1d791c3cf851970242b0bb229ce81dbada557b6d39e8c6a6a87',
    'demucs_extra.th.gz': 'f9edcf7fe55ea5ac9161c813511991e4ba03188112fd26a9135bc9308902a094',
    'light.th': '79d1ee3c1541c729c552327756954340a1a46a11ce0009dea77dc583e4b6269c',
    'light.th.gz': '94c091021d8cdee0806b6df0afbeb59e73e989dbc2c16d2c1c370b2edce774fd',
    'light_extra.th': '9e9b4af564229c80cc73c95d02d2058235bb054c6874b3cba4d5b26943a5ddcb',
    'light_extra.th.gz': '48bb1a85f5ad0ca400512fcd0dcf91ec94e886a1602a552ee32133f5e09aeae0',
    'tasnet.th': 'be56693f6a5c4854b124f95bb9dd043f3167614898493738ab52e25648bec8a2',
    'tasnet_extra.th': '0ccbece3acd98785a367211c9c35b1eadae8d148b0d37fe5a5494d6d335269b5',
}


def download_file(url, target):
    """
    Download a file with a progress bar.

    Arguments:
        url (str): url to download
        target (Path): target path to write to
        sha256 (str or None): expected sha256 hexdigest of the file
    """
    def _download():
        response = requests.get(url, stream=True)
        total_length = int(response.headers.get('content-length', 0))

        with tqdm.tqdm(total=total_length, ncols=120, unit="B", unit_scale=True) as bar:
            with open(target, "wb") as output:
                for data in response.iter_content(chunk_size=4096):
                    output.write(data)
                    bar.update(len(data))

    try:
        _download()
    except:  # noqa, re-raising
        if target.exists():
            target.unlink()
        raise


def verify_file(target, sha256):
    hasher = hashlib.sha256()
    with open(target, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hasher.update(data)
    signature = hasher.hexdigest()
    if signature != sha256:
        print(
            f"Invalid sha256 signature for the file {target}. Expected {sha256} but got "
            f"{signature}.\nIf you have recently updated the repo, it is possible "
            "the checkpoints have been updated. It is also possible that a previous "
            f"download did not run to completion.\nPlease delete the file '{target.absolute()}' "
            "and try again.",
            file=sys.stderr)
        sys.exit(1)


def encode_mp3(wav, path, verbose=False):
    try:
        import lameenc
    except ImportError:
        print("Failed to call lame encoder. Maybe it is not installed? "
              "On windows, run `python.exe -m pip install -U lameenc`, "
              "on OSX/Linux, run `python3 -m pip install -U lameenc`, "
              "then try again.", file=sys.stderr)
        sys.exit(1)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(320)
    encoder.set_in_sample_rate(44100)
    encoder.set_channels(2)
    encoder.set_quality(2)  # 2-highest, 7-fastest
    if not verbose:
        encoder.silence()
    mp3_data = encoder.encode(wav.tostring())
    mp3_data += encoder.flush()
    with open(path, "wb") as f:
        f.write(mp3_data)


def main():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    parser.add_argument("-n",
                        "--name",
                        default="demucs",
                        help="Model name. See README.md for the list of pretrained models. "
                             "Default is demucs.")
    parser.add_argument("-Q", "--quantized", action="store_true", dest="quantized", default=False,
                        help="Load the quantized model rather than the quantized version. "
                             "Quantized model is about 4 times smaller but might worsen "
                             "slightly quality.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o",
                        "--out",
                        type=Path,
                        default=Path("separated"),
                        help="Folder where to put extracted tracks. A subfolder "
                        "with the model name will be created.")
    parser.add_argument("--models",
                        type=Path,
                        default=Path("models"),
                        help="Path to trained models. "
                        "Also used to store downloaded pretrained models")
    parser.add_argument("--dl",
                        action="store_true",
                        help="Automatically download model if missing.")
    parser.add_argument("-d",
                        "--device",
                        default="cuda" if th.cuda.is_available() else "cpu",
                        help="Device to use, default is cuda if available else cpu")
    parser.add_argument("--shifts",
                        default=0,
                        type=int,
                        help="Number of random shifts for equivariant stabilization."
                        "Increase separation time but improves quality for Demucs. 10 was used "
                        "in the original paper.")
    parser.add_argument("--nosplit",
                        action="store_false",
                        default=True,
                        dest="split",
                        help="Apply the model to the entire input at once rather than "
                        "first splitting it in chunks of 10 seconds. Will OOM with Tasnet "
                        "but will work fine for Demucs if you have at least 16GB of RAM.")
    parser.add_argument("--float32",
                        action="store_true",
                        help="Convert the output wavefile to use pcm f32 format instead of s16. "
                        "This should not make a difference if you just plan on listening to the "
                        "audio but might be needed to compute exactly metrics like SDR etc.")
    parser.add_argument("--int16",
                        action="store_false",
                        dest="float32",
                        help="Opposite of --float32, here for compatibility.")
    parser.add_argument("--mp3", action="store_true",
                        help="Convert the output wavs to mp3 with 320 kb/s rate.")

    args = parser.parse_args()
    name = args.name + ".th"
    if args.quantized:
        name += ".gz"

    model_path = args.models / name
    sha256 = PRETRAINED_MODELS.get(name)
    if not model_path.is_file():
        if sha256 is None:
            print(f"No pretrained model {args.name}", file=sys.stderr)
            sys.exit(1)
        if not args.dl:
            print(
                f"Could not find model {model_path}, however a matching pretrained model exist, "
                "to download it, use --dl",
                file=sys.stderr)
            sys.exit(1)
        args.models.mkdir(exist_ok=True, parents=True)
        url = BASE_URL + name
        print("Downloading pre-trained model weights, this could take a while...")
        download_file(url, model_path)
    if sha256 is not None:
        verify_file(model_path, sha256)
    model = load_model(model_path).to(args.device)
    if args.quantized:
        args.name += "_quantized"
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    source_names = ["drums", "bass", "other", "vocals"]
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(
                f"File {track} does not exist. If the path contains spaces, "
                "please try again after surrounding the entire path with quotes \"\".",
                file=sys.stderr)
            continue
        print(f"Separating track {track}")
        wav = AudioFile(track).read(streams=0, samplerate=44100, channels=2).to(args.device)
        # Round to nearest short integer for compatibility with how MusDB load audio with stempeg.
        wav = (wav * 2**15).round() / 2**15
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav, shifts=args.shifts, split=args.split, progress=True)
        sources = sources * ref.std() + ref.mean()

        track_folder = out / track.name.split(".")[0]
        track_folder.mkdir(exist_ok=True)
        for source, name in zip(sources, source_names):
            if args.mp3 or not args.float32:
                source = (source * 2**15).clamp_(-2**15, 2**15 - 1).short()
            source = source.cpu().transpose(0, 1).numpy()
            stem = str(track_folder / name)
            if args.mp3:
                encode_mp3(source, stem + ".mp3", verbose=args.verbose)
            else:
                wavname = str(track_folder / f"{name}.wav")
                wavfile.write(wavname, 44100, source)


if __name__ == "__main__":
    main()
