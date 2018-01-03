# Adversarial Attacks against Behavioral Biometrics

This is the repository that contains the code for the paper, K-means++ vs. Behavioral Biometrics: One Loop to Rule Them All.

## Contents

  * [Setup](#setup)
      - [virtualenv](#virtualenv)
      - [Data](#data)
      - [Run](#run)
  * [Documentation](#documentation)
      - [Flags](#flags)
      - [Adding New Dataset](#adding-new-dataset)
      - [Adding New Classifier](#adding-new-classifier)
      - [Adding New Adversary](#adding-new-adversary)
      - [Selecting Samples](#selecting-samples)
      - [Pickling](#pickling)
      - [Speed Optimizations](#speed-optimizations)

  * [Acknowledgements](#acknowledgements)
  * [License](#license)


## Setup

Mostly, this just includes standard python libraries that may already be present in your system
(see requirements.txt). Here are the commands to set it up in virtualenv, or even better, use
[virtualenvwrapper](http://virtualenvwrapper.readthedocs.io/en/latest/) which manages your virtual
environments in a nicer way.

#### virtualenv

```bash
pip install virtualenv
virtualenv keystrokes
cd keystrokes
source bin/activate
git clone https://github.com/parimarjan/adversarial_keystrokes.git
cd adversarial_keystrokes
pip install -r requirements.txt
```

Note: I have not configured matplotlib with virtualenv here. If you would like to [reproduce the
graphs](#reproducing-figures), then you will need to either configure virtualenv with matplotplib,
as described [here](https://matplotlib.org/1.5.1/faq/virtualenv_faq.html), or don't use the
virtualenv, and just install the pip dependencies system wide:

#### System Wide Installation

```bash
git clone https://github.com/parimarjan/adversarial_keystrokes.git
cd adversarial_keystrokes
pip install -r requirements.txt
```

#### Data
```bash
cd datasets; sh download.sh; cd ..
```

The files and format in the dataset are described in datasets/DATASETS.md.

#### Run

If you have set everything up as described above, you should be able to just run:

```bash
python main.py --data cmu.csv --qt 1 --qtn 3 --manhattan 1 --svm 1 --rf 1 --gaussian 1 --knc 1 --gaussian_mixture 1 --fc_net 1 --cracker 1
```

This runs the Targeted K-means++ adversary on most of classifiers we used but only on 3 users -
note the parameters qt (quick test) and qtn (quick test number). This is useful to ensure
everything is working as expected for the classifiers and adversaries quickly. Note: Besides these
classifiers, we also used autoencoders (--ae), variational autoencoders (--vae), and contractive
autoencoders (--cae), but those take considerably longer to run.

To run it on the full dataset, do:

```bash
python main.py --data cmu.csv --manhattan 1 --svm 1 --cracker 1
```

which runs the Targeted K-means++ adversary on the cmu dataset with manhattan and svm classifiers.

Similarly, to run the Indiscriminate K-means++ adversary on the MTurk dataset corresponding to the
password ``mustang'', with classifiers manhattan and random forests, run:

```bash
python main.py --data mturk_password.csv --manhattan 1 --rf 1 --digraph_attack 1
```

To run this on the touchscreen swipes dataset, do:
```bash
python main.py --data android_swipes.csv --feature_norm 0 --rf 1 --cracker 1 
```

## Documentation

#### Flags

Here, I want to briefly go over some of the more useful flags and conventions used in this
repository.

* The convention for turning on a flag is, --flag 1

* Each classifier used in the run is set using the convention --classifier_name 1. By default all the
classifiers are turned off.

* --cracker 1 runs the Targeted K-means++ adversary on the given dataset and classifiers.

* --digraph_attack 1 runs the Indiscriminate K-means++ adversary on the given dataset and
classifiers.

* --feature_norm is used to turn on the [feature_normalization](https://arxiv.org/abs/1606.09075) technique
used by Vinnie Monaco. By default this is turned on, as the performance improves significantly for
the keystrokes classifier. But this technique only works for the keystrokes datasets, so for the
android touchscreen swipes, --feature_norm 0 should be used.

* --median_classifiers: This changes the acceptance threshold from the EER value, to different
percentile values based on the genuine user's scores on the training samples. It was used to
generate the Figure 3 in the NDSS paper.

* There were various other flags used to experiment with different settings. More information about
their specific use cases can be found with:

```bash
python main.py --help.
```

#### MTurk Datasets

There are five dataset files, for each of the five passwords: password, mustang, abc123, letmein,
123456789.

Note that each of the passwords in the MTurk dataset are treated as a separate dataset when running
the classifiers, and adversaries. In the paper, we presented the averaged results after running the
tests on all five password datasets. More information is given in datasets/DATASETS.md

#### Adding New Dataset

* Add the method to read in the dataset in core/loader.py, similar to the methods _read_cmu_data or
_read_swipe_data for the class Loader. In the method load_data, call the corresponding method based
on the name of your dataset. You will also need to set the value of loader.num_features in the
method to read it in.

* Format: To ensure that things run consistently later, the format should be the same as the ones
used in the keystrokes, and touchscreen swipes dataset, i.e., your read method should return a
dictionary, 'data', with each key signifying the user name, and each value a list of lists, for
each of the features of that user. For instance, data['0'][1] will be the first feature of the user
with id '0'.

* In general, this should be enough to run these classifiers and adversaries on similar datasets.
You can use the parameter self.params.dataset through the code to add special purpose code for your
particular dataset. For instance, in method run_classifier.py in core/classifiers.py, I use
self.params.android vs self.params.keystrokes vs self.params.mouse in order to use different
strategies for the test-train split of the datasets.

* This was enough to run them on the android swipes datasets, and similar mouse movements datasets
without making any changes to the keystrokes classifiers/adversaries.

#### Adding New Classifier

* Add new flag in core/parameters.py. Can do this in a way similar to the manhattan flag and so on.

* Add the classifier as a class in core/anomaly.py in the same format as:
    * One Class Classifiers: Like class Manhattan, it will need a fit and a score method.

#### Adding New Adversary

TODO: Add instructions.

#### Selecting Samples

* The way we selected training / test samples for the obtained results have been described in
detail in the paper. In short, we selected the first half of samples as training samples, and the
next half as test samples for the genuine users since that seems the most appropriate order in real
life scenarios, and it also keeps it consistent with the CMU dataset.

* The EER performance for the classifiers actually improves if we split the samples into two halves
randomly, but this doesn't seem to affect the adversarial results much. If you want to play around
with these settings, check the method run_classifier in core/classifiers.py - where the different
datasets splits have been done.

#### Pickling

Currently, we pickle the classifiers / adversaries individually based on the hash of the input
data and parameters (in self.params) which are stored in the folder /pickle. Just delete the files
in that folder to turn them off / or turn off the pickle flag with --pickle 0.

#### Speed Optimizations

Generally, the code is fairly unoptimized, and there are many places where the speed could be
improved significantly. The speed is a serious issue with the autoencoders, using the
--median_classifiers flag, and generally when using the MTurk datasets as they are rather big. A
few obvious ways to improve the runtime would be:

* Parallelization: The training loop, and the loops for attacking the users can both be
parallelized easily, as there is no dependency between the results for each user.
* Using conservative thresholds (with --median_classifiers 1) absolutely does not require retraining the
classifiers for each threshold.

## Acknowledgements

This repository has consolidated work across multiple repositories by all the paper authors. The
original hypothesis for the project was due to Dr. Bahman Bahmani, and the project was started by
Vivek Jain, who among other things implemented the K-means++ and MasterKey adversaries.

I initially began working on this project by playing around with Vinnie Monaco's excellent
keystroke dynamics [classifiers](https://github.com/vmonaco/kboc), which inspired the overall
design of this repository as well. There are also many crucial pieces of code used from Monaco's
repository in our project, and I have tried to explicitly mention this at all such places.


## Licence

TODO: Figure this out.
TODO: Add citation information.
