# Release notes for Demucs

## v2.0.0, 28th of April 2021

This is a big release, with at lof of breaking changes. You will likely
need to install Demucs from scratch.



- Demucs now supports on the fly resampling by a factor of 2.
This improves SDR almost 0.3 points.
- Random scaling of each source added (From Uhlich et al. 2017).
- Random pitch and tempo augmentation addded, from [Cohen-Hadria et al. 2019].
- With extra augmentation, the best performing Demucs model now has only 64 channels
instead of 100, so model size goes from 2.4GB to 1GB. Also SDR is up from 5.6 SDR to 6.3 when trained only on MusDB.
-  Quantized model using [DiffQ](https://github.com/facebookresearch/diffq) has been added. Model size is 150MB, no loss in quality as far as I, or the metrics,
can say.
- Pretrained models are now using the TorchHub interface.
- Overlap mode for separation, to limit inconsitencies at
	frame boundaries, with linear transition over the overlap. Overlap is currently
	at 25%. Not that this is only done for separation, not training, because
	I added that quite late to the code. For Conv-TasNet this can improve
	SDR quite a bit (+0.3 points, to 6.0).
- PyPI hosting, for separation, not training!
