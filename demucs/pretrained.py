# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import logging

from diffq import DiffQuantizer
import torch.hub

from .model import Demucs
from .tasnet import ConvTasNet
from .utils import set_state

logger = logging.getLogger(__name__)
ROOT = "https://dl.fbaipublicfiles.com/demucs/v3.0/"

PRETRAINED_MODELS = {
    'demucs': 'e07c671f',
    'demucs_extra': '3646af93',
    'demucs_quantized': '07afea75',
    'tasnet': 'beb46fac',
    'tasnet_extra': 'df3777b2',
    'demucs_unittest': '09ebc15f',
}


def get_url(name):
    sig = PRETRAINED_MODELS[name]
    return ROOT + name + "-" + sig[:8] + ".th"


def is_pretrained(name):
    return name in PRETRAINED_MODELS


def load_pretrained(name):
    if name == "demucs":
        return demucs(pretrained=True)
    elif name == "demucs_extra":
        return demucs(pretrained=True, extra=True)
    elif name == "demucs_quantized":
        return demucs(pretrained=True, quantized=True)
    elif name == "demucs_unittest":
        return demucs_unittest(pretrained=True)
    elif name == "tasnet":
        return tasnet(pretrained=True)
    elif name == "tasnet_extra":
        return tasnet(pretrained=True, extra=True)
    else:
        raise ValueError(f"Invalid pretrained name {name}")


def _load_state(name, model, quantizer=None):
    url = get_url(name)
    state = torch.hub.load_state_dict_from_url(url, map_location='cpu', check_hash=True)
    set_state(model, quantizer, state)
    if quantizer:
        quantizer.detach()


def demucs_unittest(pretrained=True):
    model = Demucs(channels=4)
    if pretrained:
        _load_state('demucs_unittest', model)
    return model


def demucs(pretrained=True, extra=False, quantized=False):
    if not pretrained and (extra or quantized):
        raise ValueError("if extra or quantized is True, pretrained must be True.")
    model = Demucs()
    if pretrained:
        name = 'demucs'
        quantizer = None
        if extra and quantized:
            raise ValueError("Only one of extra or quantized can be True.")
        if quantized:
            quantizer = DiffQuantizer(model, group_size=8, min_size=1)
            name = 'demucs_quantized'
        if extra:
            name = 'demucs_extra'
        _load_state(name, model, quantizer)
    return model


def tasnet(pretrained=True, extra=False):
    if not pretrained and extra:
        raise ValueError("if extra is True, pretrained must be True.")
    model = ConvTasNet(X=10)
    if pretrained:
        name = 'tasnet'
        if extra:
            name = 'tasnet_extra'
        _load_state(name, model)
    return model
