# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import hashlib
import sys
import warnings
from pathlib import Path

import requests
import torch as th
import tqdm
from scipy.io import wavfile

from .audio import AudioFile
from .utils import apply_model

BASE_URL = "https://dl.fbaipublicfiles.com/demucs/"
PRETRAINED_MODELS = {
    'demucs': 'b25b32a4772e43983d2e30cce17a85aaa1baf3aea0a6145eb591b0f393ac9312',
    'demucs_extra': '099d3e7fcbb8da7484184ad6d84cd6c3820947ff0e2f52d5e622365685037cd7',
    'tasnet': 'd037cdbad294f1a7152af8742d8419c305e244810381a795ffcd22b8593360d8',
    'tasnet_extra': '4a91ade51439ac41bb1914753c3aebd839eed17981379e6142478bcfa3a5461f',
}


def download_file(url, target, sha256=None):
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

        if sha256 is not None:
            hasher = hashlib.sha256()

        with tqdm.tqdm(total=total_length, unit="B", unit_scale=True) as bar:
            with open(target, "wb") as output:
                for data in response.iter_content(chunk_size=4096):
                    output.write(data)
                    if sha256 is not None:
                        hasher.update(data)
                    bar.update(len(data))
        if sha256 is not None:
            signature = hasher.hexdigest()
            if sha256 != signature:
                raise ValueError("Invalid sha256 signature when downloading {}. "
                                 "Expected {} but got {}".format(url, sha256, signature))

    try:
        _download()
    except:  # noqa, re-raising
        if target.exists():
            target.unlink()
        raise


def main():
    parser = argparse.ArgumentParser("demucs.separate",
                                     description="Separate the sources for the given tracks")
    parser.add_argument("tracks", nargs='+', type=Path, default=[], help='Path to tracks')
    parser.add_argument("-n",
                        "--name",
                        default="demucs",
                        help="Model name. See README.md for the list of pretrained models.")
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
                        "first splitting it in chunks of 16 seconds. Will OOM with Tasnet "
                        "but will work fine for Demucs if you have at least 16GB of RAM.")

    args = parser.parse_args()
    model_path = args.models / f"{args.name}.th"
    if not model_path.is_file():
        sha256 = PRETRAINED_MODELS.get(args.name)
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
        url = BASE_URL + f"{args.name}.th"
        print("Downloading pre-trained model weights, this could take a while...")
        download_file(url, model_path, sha256)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = th.load(model_path).to(args.device)
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
            source = source.cpu().transpose(0, 1).numpy()
            wavfile.write(str(track_folder / f"{name}.wav"), 44100, source)


if __name__ == "__main__":
    main()
