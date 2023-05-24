# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import warnings
from pathlib import Path

from dora.log import fatal
import torch as th

from .api import Separator, save_audio

from .apply import BagOfModels
from .htdemucs import HTDemucs
from .pretrained import add_model_flags, ModelLoadingError


def get_parser():
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
    parser.add_argument("--filename",
                        default="{track}/{stem}.{ext}",
                        help="Set the name of output file. \n"
                        'Use "{track}", "{trackext}", "{stem}", "{ext}" to use '
                        "variables of track name without extension, track extension, "
                        "stem name and default output file extension. \n"
                        'Default is "{track}/{stem}.{ext}".')
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
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--no-split",
                             action="store_false",
                             dest="split",
                             default=True,
                             help="Doesn't split audio in chunks. "
                             "This can use large amounts of memory.")
    split_group.add_argument("--segment", type=int,
                             help="Set split size of each chunk. "
                             "This can help save memory of graphic card. ")
    parser.add_argument("--two-stems",
                        dest="stem", metavar="STEM",
                        help="Only separate audio into {STEM} and no_{STEM}. ")
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument("--int24", action="store_true",
                             help="Save wav output as 24 bits wav.")
    depth_group.add_argument("--float32", action="store_true",
                             help="Save wav output as float32 (2x bigger).")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp"],
                        help="Strategy for avoiding clipping: rescaling entire signal "
                             "if necessary  (rescale) or hard clipping (clamp).")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument("--flac", action="store_true",
                              help="Convert the output wavs to flac.")
    format_group.add_argument("--mp3", action="store_true",
                              help="Convert the output wavs to mp3.")
    parser.add_argument("--mp3-bitrate",
                        default=320,
                        type=int,
                        help="Bitrate of converted mp3.")
    parser.add_argument("-j", "--jobs",
                        default=0,
                        type=int,
                        help="Number of jobs. This can increase memory usage but will "
                             "be much faster when multiple cores are available.")

    return parser


def main(opts=None):
    parser = get_parser()
    args = parser.parse_args(opts)

    try:
        separator = Separator()
        separator.load_model(model=args.name, repo=args.repo)
    except ModelLoadingError as error:
        fatal(error.args[0])

    max_allowed_segment: float = float('inf')
    if isinstance(separator.model, HTDemucs):
        max_allowed_segment = float(separator.model.segment)
    elif isinstance(separator.model, BagOfModels):
        max_allowed_segment = separator.model.max_allowed_segment
    if args.segment is not None and args.segment > max_allowed_segment:
        fatal("Cannot use a Transformer model with a longer segment "
              f"than it was trained for. Maximum segment is: {max_allowed_segment}")

    if isinstance(separator.model, BagOfModels):
        print(
            f"Selected model is a bag of {len(separator.model.models)} models. "
            "You will see that many progress bars per track."
        )

    if args.stem is not None and args.stem not in separator.model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. '
            "STEM must be one of {sources}.".format(
                stem=args.stem, sources=", ".join(separator.model.sources)
            )
        )
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Separated tracks will be stored in {out.resolve()}")
    for track in args.tracks:
        if not track.exists():
            print(f"File {track} does not exist. If the path contains spaces, "
                  'please try again after surrounding the entire path with quotes "".',
                  file=sys.stderr)
            continue
        print(f"Separating track {track}")
        separator.clear_filelist()
        with warnings.catch_warnings(record=True) as w:
            separator.load_audios_to_model(track)
            if len(w):
                print(f"Failed to load {track}. Skipped.", file=sys.stderr)
                continue

        res = next(
            iter(
                separator.separate_loaded_audio(
                    device=args.device,
                    shifts=args.shifts,
                    split=args.split,
                    overlap=args.overlap,
                    progress=True,
                    num_workers=args.jobs,
                    segment=args.segment,
                )
            )
        )[1]

        if args.mp3:
            ext = "mp3"
        elif args.flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": args.mp3_bitrate,
            "clip": args.clip_mode,
            "as_float": args.float32,
            "bits_per_sample": 24 if args.int24 else 16,
        }
        if args.stem is None:
            for name, source in res.items():
                stem = out / args.filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem=name,
                    ext=ext,
                )
                stem.parent.mkdir(parents=True, exist_ok=True)
                save_audio(source, str(stem), **kwargs)
        else:
            sources = list(res.values())
            stem = out / args.filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem=args.stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(sources.pop(separator.model.sources.index(args.stem)),
                       str(stem), **kwargs)
            # Warning : after poping the stem, selected stem is no longer in the list 'sources'
            other_stem = th.zeros_like(sources[0])
            for i in sources:
                other_stem += i
            stem = out / args.filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem="no_" + args.stem,
                ext=ext,
            )
            stem.parent.mkdir(parents=True, exist_ok=True)
            save_audio(other_stem, str(stem), **kwargs)


if __name__ == "__main__":
    main()
