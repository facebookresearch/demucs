import sys

from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np
from torchaudio.utils import download_asset
import torchaudio


model_name = "htdemucs"

with httpclient.InferenceServerClient("localhost:8000") as client:

    SAMPLE_SONG = download_asset("tutorial-assets/hdemucs_mix.wav")
    waveform, sample_rate = torchaudio.load(SAMPLE_SONG)
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    waveform = waveform[None]

    input_data = waveform.numpy().astype(np.float32)
    inputs = [
        httpclient.InferInput("INPUT", input_data.shape,
                              np_to_triton_dtype(input_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT"),
    ]

    response = client.infer(model_name=model_name,
                            inputs=inputs,
                            request_id=str(1),
                            outputs=outputs)

    result = response.get_response()
    output_data = response.as_numpy("OUTPUT")

    print("INPUT ({}) => OUTPUT ({})".format(
        input_data, output_data))

    print('PASS: htdemucs')
    sys.exit(0)
