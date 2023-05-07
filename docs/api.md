# Demucs APIs

## Quick start

1. The first step is to import api module:

```python
import demucs.api
```

2. Then initialize the `Separator`. It can either use a command line, or use keyword arguments, or both. The initialize option will do nothing except registering the class. 

```python
# Initialize with no parameters:
separator = demucs.api.Separator()

# Initialize with command line:
separator = demucs.api.Separator(["-n", "mdx_extra", "--segment=12"])

# Initialize with command line (argv) and add keyword arguments:
import sys
separator = demucs.api.Separator(sys.argv[1:], name="mdx_extra")
```

3. Load a model

```python
# If running without parameter, the arguments used to initialize will be used
separator.load_model()

# Or select a model and receive it
model = separator.load_model(name="htdemucs_ft", repo="./pretrained")
```

4. Load tracks (into the separator if you like, which is recommended)

```python
# Load audios
audios = ["1.mp3", "2.ogg", "3.flac"]
separator.load_audios_to_model(*audios)

# Load the audios that is specified in the command line:
separator.load_audios_from_cmdline()
```

5. Separate it!

```python
# Separating loaded audios with argument specified in command line
separated = separator.separate_loaded_audio()
```

6. Save audio

```python
for file, sources in separated:
    for stem, source in sources.items():
        demucs.api.save_audio(source, f"{stem}_{file}", samplerate=separator._samplerate)
```

## API References

The types of each parameter and return value is not listed in this document. To know the exact type of them, please read the type hints in api.py (most modern code editors support infering types based on type hints).

### `class Separator`

The base separator class

##### Parameters

cmd_line (optional): Parsed command line, use `sys.argv[1:]` to use runtime command line. Supported commands are same as `demucs.separate`. Please remember that not all the options are supported.

kw: Arguments to be added or replaced in the command line. To get a list of supported keys, you can run `print(demucs.separate.get_parser().parse_args([""]))`.

#### `property samplerate`

A read-only property saving sample rate of the model requires. Will raise a warning if the model is not loaded and return the default value.

#### `property audio_channels`

A read-only property saving audio channels of the model requires. Will raise a warning if the model is not loaded and return the default value.

#### `property model`

A read-only property saving the model.

#### `property model`

A read-only property saving the parsed arguments.

#### `method load_model()`

Load a model to the class and return the model. This could only be called once.

To manually add a loaded model to the class, simply assign the `Separator._model` variable.

##### Parameters

model: If not specified, will use the model specified in the command line. 

repo: If not specified, will use the model specified in the command line.

##### Returns

Model (Demucs | HDemucs | HTDemucs | BagOfModels)

#### `method load_audios()`

Load several audios and return the iterator of the audios. This function returns an iterator, so the audio will not be decoded and read into memory until the iterator reached the spcefic item.

If you want to just read one audio, you can use `next(iter(Separator.load_audios(path)))`

If you want to load several audio at once, please use `list(Separator.load_audios(path))`

##### Parameters

tracks: Pathlike objects, containing the path of the audio to be loaded.

audio_channels: The targeted audio channels. If not specified, will use the value of the loaded model. If no model is loaded, 2 is the default value.

samplerate: The targeted audio channels. If not specified, will use the value of the loaded model. If no model is loaded, 44100 is the default value.

ignore_errors: If true, any exception encountered will be ignored and the audio failed to be loaded will become `None`.

##### Returns

A generator (iterator) of tuple[filename, wave]. If `ignore_errors` is True (default), the wave of that file will be `None`.

#### `method load_audios_to_model()`

Load several audios to the Separator that can be used in `Separator.separate_loaded_audio`.

##### Parameters

tracks: Pathlike objects, containing the path of the audio to be loaded.

##### Returns

None

##### Notes

When an error encountered, this function will only warn and continue loading other audios. The audio failed to load will not be added into the Separator. To get a list of audios failed to be loaded, you can use the following codes:

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    Separator.load_audios_to_model(track)
    failures = list(i.message.separate('"')[1] for i in w)
```

#### `method load_audios_from_cmdline()`

Load audios specied in the command line to the Separator that can be used in `Separator.separate_loaded_audio`.

##### Parameters

None

##### Returns

None

##### Notes

When an error encountered, this function will only warn and continue loading other audios. The audio failed to load will not be added into the Separator. To get a list of audios failed to be loaded, you can use the following codes:

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    Separator.load_audios_from_cmdline()
    failures = list(i.message.separate('"')[1] for i in w)
```

#### `method clear_filelist()`

Remove all the loaded audios in the Separator.

##### Parameters

None

##### Returns

None

#### `method add_track()`

Add a loaded track into the separator.

##### Parameters

filename: A string for you to remember what the each audio is. (You can call it "identifier")

wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, while the second is the waveform of each channel. e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.

##### Returns

None

##### Notes

Use this function with cautiousness. This function does not provide data verifying.

#### `method separate_audio()`

Separate an audio.

##### Parameters

wav: Waveform of the audio. Should have 2 dimensions, the first is each audio channel, while the second is the waveform of each channel. e.g. `tuple(wav.shape) == (2, 884000)` means the audio has 2 channels.

model: Model to be used. If not specified, will use the model loaded to the Separator.

segment: Length (in seconds) of each segment (only available if `split` is `True`). If not specified, will use the command line option.

shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and apply the oppositve shift to the output. This is repeated `shifts` time and all predictions are averaged. This effectively makes the model time equivariant and improves SDR by up to 0.2 points. If not specified, will use the command line option.

split: If True, the input will be broken down into small chunks (length set by `segment`) and predictions will be performed individually on each and concatenated. Useful for model with large memory footprint like Tasnet. If not specified, will use the command line option.

overlap: The overlap between the splits. If not specified, will use the command line option.

device (torch.device, str, or None): If provided, device on which to execute the computation, otherwise `wav.device` is assumed. When `device` is different from `wav.device`, only local computations will be on `device`, while the entire tracks will be stored on `wav.device`. If not specified, will use the command line option.

num_workers: Number of jobs. This can increase memory usage but will be much faster when multiple cores are available. If not specified, will use the command line option.

callback: A function will be called when the separation of a chunk starts or finished. The argument passed to the function will be a dict. For more information, please see the Callback section.

callback_arg: A dict containing private parameters to be passed to callback function. For more information, please see the Callback section.

##### Returns

Separated stems.

##### Callback

The function will be called with only one positional parameter whose type is `dict`. The `callback_arg` will be combined with information of current separation progress. The progress information will override the values in `callback_arg` if same key has been used.

Progress information contains several keys (These keys will always exist):
- `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
- `shift_idx`: The index of shifts. Starts from 0.
- `segment_offset`: The offset of current segment. If the number is 441000, it doesn't mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
- `state`: Could be `"start"` or `"end"`.
- `audio_length`: Length of the audio (in "frame" of the tensor).
- `models`: Count of submodels in the model.

##### Notes

Use this function with cautiousness. This function does not provide data verifying.

#### `method separate_loaded_audio()`

Separate the audio loaded into the Separator.

##### Parameters

segment: Length (in seconds) of each segment (only available if `split` is `True`). If not specified, will use the command line option.

shifts: If > 0, will shift in time `wav` by a random amount between 0 and 0.5 sec and apply the oppositve shift to the output. This is repeated `shifts` time and all predictions are averaged. This effectively makes the model time equivariant and improves SDR by up to 0.2 points. If not specified, will use the command line option.

split: If True, the input will be broken down into small chunks (length set by `segment`) and predictions will be performed individually on each and concatenated. Useful for model with large memory footprint like Tasnet. If not specified, will use the command line option.

overlap: The overlap between the splits. If not specified, will use the command line option.

device (torch.device, str, or None): If provided, device on which to execute the computation, otherwise `wav.device` is assumed. When `device` is different from `wav.device`, only local computations will be on `device`, while the entire tracks will be stored on `wav.device`. If not specified, will use the command line option.

num_workers: Number of jobs. This can increase memory usage but will be much faster when multiple cores are available. If not specified, will use the command line option.

callback: A function will be called when the separation of a chunk starts or finished. The argument passed to the function will be a dict. For more information, please see the Callback section.

callback_arg: A dict containing private parameters to be passed to callback function. For more information, please see the Callback section.

##### Returns

A generator (iterator) of tuple[filename, dict[stem_name, waveform]].

##### Callback

The function will be called with only one positional parameter whose type is `dict`. The `callback_arg` will be combined with information of current separation progress. The progress information will override the values in `callback_arg` if same key has been used.

Progress information contains several keys (These keys will always exist):
- `model_idx_in_bag`: The index of the submodel in `BagOfModels`. Starts from 0.
- `shift_idx`: The index of shifts. Starts from 0.
- `segment_offset`: The offset of current segment. If the number is 441000, it doesn't mean that it is at the 441000 second of the audio, but the "frame" of the tensor.
- `state`: Could be `"start"` or `"end"`.
- `audio_length`: Length of the audio (in "frame" of the tensor).
- `models`: Count of submodels in the model.
- `file`: File name (or what you may call "identifier") of the waveform that is being separated

##### Notes

The returns of the function is an iterator, so the separation process will not start until the iterator reaches the specific item.

To separate the first audio in the loaded list, use `next(iter(Separator.separate_loaded_audio()))`.

To separate all the audio at once (which may consume lots of memory), use `list(Separator.separate_loaded_audio())`.

The function will not remove the separated tracks from the Separator. So run `Separator.clear_filelist()` if you like to remove them.

To get the separated tracks, just retrieve `Separator._out` variable. It will be cleared each time you run this function.

### `function save_audio()`

Save audio file.

##### Parameters

wav: Audio to be saved

path: The file path to be saved. Ending must be one of `.mp3` and `.wav`.

samplerate: File sample rate.

bitrate: If the suffix of `path` is `.mp3`, it will be used to specify the bitrate of mp3.

clip: Clipping preventing strategy.

bits_per_sample: If the suffix of `path` is `.wav`, it will be used to specify the bit depth of wav.

as_float: If it is True and the suffix of `path` is `.wav`, then `bits_per_sample` will be set to 32 and will write the wave file with float format.

##### Returns

None
