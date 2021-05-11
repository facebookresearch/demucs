# Music DemiXing challenge (MDX)

If you want to use Demucs for the [MDX challenge](https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021),
please follow the instructions hereafter

## Installing Demucs

Follow the instructions from the [main README](https://github.com/facebookresearch/demucs#requirements)
in order to setup Demucs using Anaconda. You will need the full setup up for training, including soundstretch.

## Getting MusDB-HQ

Download [MusDB-HQ](https://zenodo.org/record/3338373) to some folder and unzip it.

## Training Demucs

Train Demucs (you might need to change the batch size depending on the number of GPUs available).
It seems 48 channels is enough to get the best performance on MusDB-HQ, and training will faster
and less memory demanding.
```bash
./run.py --channels=48 --batch_size 64 --musdb=PATH_TO_MUSDB --is_wav [EXTRA_FLAGS]
```

Once the training is completed, a new model file will be exported in `models/`.
You can look at the SDR on the MusDB dataset using `python result_table.py`.


### Evaluate and export a model before training is over

If you want to export a model before training is complete, use the following command:
```bash
python -m demucs [ALL EXACT TRAINING FLAGS] --save_model
```
Once this is done, you can partially evaluate a model with
```bash
./run.py --test models/NAME_OF_MODEL.th --musdb=PATH_TO_MUSDB --is_wav
```

## Submitting your model

Git clone [the Music Demixing Challenge - Starter Kit](https://github.com/AIcrowd/music-demixing-challenge-starter-kit).
In this repository, change the `predict.py` to use the `DemucsSeparator`.
Inside the starter kit, create a `models/` folder and copy over the trained model from the Demucs repo (renaming
it for instance `my_model.th`)
Inside the `test_demuc.py` file, change the function `prediction_setup`: comment the loading
of the pre-trained model, and uncomment the code to load your own model.


Install [git-lfs](https://git-lfs.github.com/). Then run

```bash
git lfs install
git lfs track models/my_model.th
git add .gitattributes
git add models/
git add -u .
git commit -m "My Demucs submission"
```
and then follow the [submission instructions](https://github.com/AIcrowd/music-demixing-challenge-starter-kit/blob/master/docs/SUBMISSION.md).

Best of luck!
