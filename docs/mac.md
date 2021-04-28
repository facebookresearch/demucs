# Mac OS X support for Demucs

If you have a sufficiently recent version of OS X, you can just run

```bash
python3 -m pip install --user -U demucs
# Then anytime you want to use demucs, just do
python3 -m demucs.separate -d cpu PATH_TO_AUDIO_FILE_1
```

If you do not already have Anaconda installed or much experience with the terminal on Mac OS X here are some detailed instructions:

1. Download [Anaconda 3.8 (or more recent) 64 bits for MacOS][anaconda]:
2. Open [Anaconda Prompt in MacOSX][prompt]
3. Follow these commands:
```bash
conda activate
pip3 install -U demucs
# Then anytime you want to use demucs, first do conda activate, then
python3 -m demucs.separate -d cpu PATH_TO_AUDIO_FILE_1
```

[anaconda]:  https://www.anaconda.com/distribution/#download-section
[prompt]: https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-nav-mac