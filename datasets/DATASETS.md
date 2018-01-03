# Datasets

## Contents
  * [Downloading Datasets](#download)
  * [Description](#description)
    - [CMU Dataset](#cmu-dataset)
    - [MTurk Datasets](#mturk-datasets)
    - [Android Swipes Dataset](#android-swipes-dataset)
    - [Indiscriminate Adversarial Samples](#indiscriminate-adversarial-samples)
  * [Data Collection Details](#details)


## Download
To download all the datasets, just run the shell script:

```bash
$ sh download.sh
```
Or you can just download each of the datasets individually based on the script.

## Description
All the datasets follow the format used by the [Killourhy and Maxion CMU dataset](https://www.cs.cmu.edu/~keystroke/)

#### CMU Dataset

This is the standard benchmark dataset used for a long time in many keystroke dynamics research
papers. It has been described in detail [here](https://www.cs.cmu.edu/~keystroke/).

#### MTurk Datasets

##### Basic Details
This dataset contains samples of the following five passwords:

1. password
2. mustang
3. abc123
4. letmein
5. 123456789

Each were typed a 100 times by users on MTurk in a single session. We treat each password as a
'new' dataset - with the same format as the CMU dataset. To produce the results in the paper, we
ran the classifiers and adversaries on each dataset individually.

##### Data Format

This follows the exact same format as the [CMU dataset](https://www.cs.cmu.edu/~keystroke/#sec2).
Each row represents a new repetition of the password being typed by the user. 

The first few columns are:

* user: A randomized, unique id for this user. These are consistent across the five password datasets.
* session: Since there was only one session, this is always 1.
* rep: repetition, ranges from 1 to 100.

After this, we have the timing values in milliseconds. It follows the CMU dataset having
the following three features per keystroke:
* Hold Time: how long was the current key held down for. e.g., In mturk_password.csv, the first
timing value represents duration of 'p' being held down.
* Down Down: the time from pressing the first key down, to pressing the next key down. e.g., In
mturk_password.csv, the second timing value represents time between 'p' being pressed, and 'a'
being pressed.
* Up Down: the time from releasing the first key, to pressing the next key. e.g., In
mturk_password.csv, the third timing represents time between 'p' being released, to 'a' being
pressed. Note that this can also be negative at times.

In reality, only two of these values are needed as the third can be derived from any two, but we
wanted to keep the format / and analysis consistent with the CMU dataset.

So each character, instead of the last one, has three timing values associated with them, and the
last one has just one (hold time). These columns don't have any headers, but they can be easily
inferred based on the above description.

##### How were these passwords selected?

We took these from one of the lists for the most common passwords. One of the ideas behind
collecting this dataset was to use common English words so the users would not struggle to get used
to it.

##### Website for collecting the dataset

TODO: check with Vivek and make the website code public.

##### Why was this dataset collected?

TODO: Copy from the paper text.

###### Is this dataset representative despite not containing any special characters?

Adding special characters, or capitralized characters, shouldn't really change anything because each
key - like the shift key used for many special characters - is just treated as a new keypress as in
the cmu dataset. 

###### Raw data and cleaning up script

Some of the raw data was lost or corrupted after the experiment. 

TODO: Add exact stats of the users who were dropped.
TODO: Mention some of the cleaning up techniques that we used.

#### Android Swipes Dataset
Details about this dataset are provided [here](http://www.ms.sapientia.ro/~manyi/personality.html).

#### Indiscriminate Adversarial Samples

Here, we used samples from other users typing different words in order to construct attacks against
the five passwords in the MTurk dataset described above. This simulates a scenario in which an
adversary might have a large corpus of typing data from the general population. If he had enough
samples for each possible digraph pair of key-presses (and there aren't that many keys on the
keyboard), then he could construct attack timings as in our paper against any password.

##### Format

Here we provide json dictionary which contain all the timings for each of the keypresses. The keys
representing either the hold time for a single key, like key 'p', or the time between a digraph, like
key 'pa'. If you would like to get the data for the full words being typed, just contact us.

##### Words Used

TODO: Update list from paper.

###### mustang

* mu: mumble, mutter
* us:
* st:
* ta:
* an:
* ng:

###### password

* pa: pat, part
* as: taste, fast
* ss: boss, cross
* sw: swat, answer
* wo: woman, wolf
* or: bored, more
* rd: shard, gird

##### letmein
* le: lest, leak
* et: beta, met
* tm: paytm, tmux
* me: me, same
* ei: veil, height
* in: win, sin

##### abc123
* ab: 'fab', 'abs'
* bc: 'bobcat', 'bc'
* c1: 'mac1', 'c1m'
* 12: '2312', '42312'
* 23: '235', '423'

##### 123456789

* just used random numbers which overlapped with each pair here.
