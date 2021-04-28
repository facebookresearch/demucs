# Windows support for Demucs

If you are using Windows, replace `python3` by `python.exe` in all the commands provided hereafter :)

Parts of the code are untested on Windows (in particular, training a new model). If you don't have much experience with Anaconda, python or the shell, here are more detailed instructions. Note that **Demucs is not supported on 32bits systems** (as Pytorch is not available there).

- First install Anaconda with **Python 3.7** or more recent, which you can find [here][install].
- Start the [Anaconda prompt][prompt].
- Type in the following commands:

```bash
cd %HOMEPATH%
python.exe -m pip install -U demucs
python.exe -m demucs.separate -d cpu "PATH_TO_AUDIO_FILE_1" ["PATH_TO_AUDIO_FILE_2" ...]
```
The `"` around the filename are required if the path contains spaces.
The separated files will be under `C:\Users\YOUR_USERNAME\demucs\separated\demucs\`. The next time you want to use Demucs, start again the [Anaconda prompt][prompt] and simply run
```bash
cd %HOMEPATH%
cd demucs
python.exe -m demucs.separate -d cpu "PATH_TO_AUDIO_FILE_1" ...
```

If you have an error saying that `mkl_intel_thread.dll` cannot be found, you can try to first run
`conda install -c defaults intel-openmp -f`. Then try again to run the `demucs.separate` command. If it still doesn't work, you can try to run first `set CONDA_DLL_SEARCH_MODIFICATION_ENABLE=1`, then again the `demucs.separate` command and hopefully it will work 🙏.
If you get a permission error, please try starting the Anaconda Prompt as administrator.


**If you are running a 32bit version of Windows**: PyTorch won't be supported, so you cannot run Demucs.

[install]: https://www.anaconda.com/distribution/#windows
[prompt]: https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-prompt-win
