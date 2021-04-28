# Linux support for Demucs

If your distribution has at least Python 3.7, and you just wish to separate
tracks with Demucs, not train it, you can just run

```bash
pip3 install --user -U demucs
# Then anytime you want to use demucs, just do
python3 -m demucs.separate -d cpu PATH_TO_AUDIO_FILE_1
```

If Python is too old, or you want to be able to train, I recommend [installing Miniconda][miniconda], with Python 3.7 or more.

```bash
conda activate
python3 install -U demucs
# Then anytime you want to use demucs, first do conda activate, then
python3 -m demucs.separate -d cpu PATH_TO_AUDIO_FILE_1
```

Of course, you can also use a specific env for Demucs.


[miniconda]: https://docs.conda.io/en/latest/miniconda.html#linux-installers