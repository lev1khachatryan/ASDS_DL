# Simple Audio Recognition

This tutorial will show you how to build a basic speech recognition network that recognizes ten different words. It's important to know that real speech and audio recognition systems are much more complex, but like MNIST for images, it should give you a basic understanding of the techniques involved. Once you've completed this tutorial, you'll have a model that tries to classify a one second audio clip as either silence, an unknown word, "yes", "no", "up", "down", "left", "right", "on", "off", "stop", or "go". You'll also be able to take this model and run it in an Android application.

## Preparation

You should make sure you have TensorFlow installed, and since the script downloads over 1GB of training data, you'll need a good internet connection and enough free space on your machine. The training process itself can take several hours, so make sure you have a machine available for that long.

## Training

To begin the training process, go to the source tree and run:

```{python}
python execute_notebook.py
```

## Confusion Matrix

```
I0730 16:57:38.073667   55030 train.py:243] Confusion Matrix:
 [[258   0   0   0   0   0   0   0   0   0   0   0]
 [  7   6  26  94   7  49   1  15  40   2   0  11]
 [ 10   1 107  80  13  22   0  13  10   1   0   4]
 [  1   3  16 163   6  48   0   5  10   1   0  17]
 [ 15   1  17 114  55  13   0   9  22   5   0   9]
 [  1   1   6  97   3  87   1  12  46   0   0  10]
 [  8   6  86  84  13  24   1   9   9   1   0   6]
 [  9   3  32 112   9  26   1  36  19   0   0   9]
 [  8   2  12  94   9  52   0   6  72   0   0   2]
 [ 16   1  39  74  29  42   0   6  37   9   0   3]
 [ 15   6  17  71  50  37   0   6  32   2   1   9]
 [ 11   1   6 151   5  42   0   8  16   0   0  20]]
```

The first section is a confusion matrix. To understand what it means, you first need to know the labels being used, which in this case are "silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go". Each column represents a set of samples that were predicted to be each label, so the first column represents all the clips that were predicted to be silence, the second all those that were predicted to be unknown words, the third "yes", and so on.

Each row represents clips by their correct, ground truth labels. The first row is all the clips that were silence, the second clips that were unknown words, the third "yes", etc.

This matrix can be more useful than just a single accuracy score because it gives a good summary of what mistakes the network is making. In this example you can see that all of the entries in the first row are zero, apart from the initial one. Because the first row is all the clips that are actually silence, this means that none of them were mistakenly labeled as words, so we have no false negatives for silence. This shows the network is already getting pretty good at distinguishing silence from words.

If we look down the first column though, we see a lot of non-zero values. The column represents all the clips that were predicted to be silence, so positive numbers outside of the first cell are errors. This means that some clips of real spoken words are actually being predicted to be silence, so we do have quite a few false positives.

A perfect model would produce a confusion matrix where all of the entries were zero apart from a diagonal line through the center. Spotting deviations from that pattern can help you figure out how the model is most easily confused, and once you've identified the problems you can address them by adding more data or cleaning up categories.

