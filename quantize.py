# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Quantize a pre-trained model. Just pass the path to the model to this script
and it will save a gzipped compressed version of the model with the weights quantized
over 8 bits. The model is still stored as floats, but gzip finds out on it own
that only 256 different float values exist and do the compression for us.
"""
import sys

from demucs.utils import load_model, save_model


def quantize(p, level=256):
    scale = p.abs().max()
    fac = 2 * scale / (level - 1)
    q = ((p + scale) / fac).round()
    p = q * fac - scale
    return p


def main():
    path = sys.argv[1]
    level = 256
    min_mb = 20
    if len(sys.argv) >= 3:
        level = int(sys.argv[2])
    if len(sys.argv) >= 4:
        min_mb = float(sys.argv[3])

    print(path, level, min_mb)
    model = load_model(path)
    for p in model.parameters():
        if p.numel() >= min_mb * 2**20 / 4:
            p.data[:] = quantize(p.data, level)
    save_model(model, path + ".gz")


if __name__ == "__main__":
    main()
