# Windows support for Demucs

## Installation and usage

Parts of the code are untested on Windows (in particular, training a new model). If you don't have much experience with Anaconda, python or the shell, here are more detailed instructions. Note that **Demucs is not supported on 32bits systems** (as Pytorch is not available there).

- First install Anaconda with **Python 3.7** or more recent, which you can find [here][install].
- Start the [Anaconda prompt][prompt].

Then, all commands that follow must be run from this prompt.

### If you want to use your GPU

If you have graphic cards produced by nVidia with more than 6GiB of memory, you can separate tracks with GPU acceleration. To achieve this, you must install Pytorch with CUDA. If Pytorch was already installed (you already installed Demucs for instance), first run  `python.exe -m pip uninstall torch torchaudio`.
Then visit [Pytorch Home Page](https://pytorch.org/get-started/locally/) and follow the guide on it to install with CUDA support. 

### Installation

Start the Anaconda prompt, and run the following
bash
```
conda install -c conda-forge ffmpeg
python.exe -m pip install -U demucs PySoundFile
```

### Upgrade

To upgrade Demucs, simply run `python.exe -m pip install -U demucs`, from the Anaconda prompt.

### Usage

Then to use Demucs, just start the **Anaconda prompt** and run:
```
demucs -d cpu "PATH_TO_AUDIO_FILE_1" ["PATH_TO_AUDIO_FILE_2" ...]
```
The `"` around the filename are required if the path contains spaces.
The separated files will be under `C:\Users\YOUR_USERNAME\demucs\separated\demucs\`.


### Separating an entire folder

You can use the following command to separate an entire folder of mp3s for instance (replace the extension `.mp3` if needs be for other file types)
```
cd FOLDER
for %i in (*.mp3) do (demucs -d cpu "%i")
```


## Potential errors

If you have an error saying that `mkl_intel_thread.dll` cannot be found, you can try to first run
`conda install -c defaults intel-openmp -f`. Then try again to run the `demucs` command. If it still doesn't work, you can try to run first `set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`, then again the `demucs` command and hopefully it will work 🙏.

**If you get a permission error**, please try starting the Anaconda Prompt as administrator.


[install]: https://www.anaconda.com/distribution/#windows
[prompt]: https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-prompt-win
