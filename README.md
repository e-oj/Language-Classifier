# Language Classifier
## Language classification using decision trees and adaboost.

This classifier distinguishes between two or more languages. It's currently set up to differentiaite English from 
Dutch but you can classify other languages by modifying the training data. You can also edit the language features to get 
stronger results for new languages. For extra details, checkout this [writeup](https://docs.google.com/document/d/1TWwhFmji458pAycIzHSXn9rB8dsC8AZpyY7Qghsrwew/edit?usp=sharing). <b>This program has no dependencies. Every algorithm used was implemented from scratch.</b>

<br>

### Algorithms
Two classification methods are used by this program.

#### Decision Tree:
A decision tree is built from the training data. Entropy is used as a measure of impurity in a given set, and the Information gain algorithm is used to split the data by features. The decision can be assigned a maximum depth to restrict its growth.

<br>

### Usage
#### The program has two entry points in the root directory, accessible via the following commands:

<b>python3 classify.py train</b> `<examples>` `<hypothesisOut>` `<learning-type>`
 - `<examples>` is a file containing labeled examples. For example.
 - `<hypothesisOut>` specifies the filename to write your model to.
 - `<learning-type>` specifies the type of learning algorithm you will run, it is either "dt" or "ada".


<b>python3 classify.py predict</b> `<hypothesis>` `<file>`
 - `<hypothesis>` is a pre-trained model created by your train program
 - `<file>` is a file containing test data.

<br>

### Training and Test Data
Here's the format for training and test data

```
<label>|<data>
```

For English and Dutch, the labels ```en``` and ```nl``` are used, respectively. Here's an exerpt from the test file:

```
en|two percussionists and a string quartet) performs six of the leader's originals and, although none
nl|maakte hij zijn debuut op het hoogste niveau. Schena mocht dertien keer meespelen in het
en|1984, where the then-World Wrestling Federation put him over the likes of Johnny Rodz
nl|Trophy tekende hij een contract in de I-League bij regerend kampioen Dempo SC voor het
en|Eventually, five princes came to Taketori no Okina's residence to ask for the beautiful Kaguya-hime's
nl|Romeinen was en het feit dat er van hem geen Germaanse naam is overgeleverd, is
```

The full test data can be found in the `/in/test.dat` file. Training data can be found at `/in/train.dat` .

<br>

### Pre-trained Models
Trained models that classify English and Dutch can be found in the `\out` directory.

 - `\out\ensemble.oj` contains an adaboost ensemble.
 - `\out\tree.oj` contains a decision tree.
 
Any of these files can be used to run the classification job.

<br>

### Language Features
The strength of a model is heavily dependent on the strength of the features used in training. While some were generic, 
a vast majority of the features used were geared towards English and Dutch. You can modify these features in the 
```get_features``` function of ```instance.py```. A more detailed explanation of the features can be found 
[in the writeup](https://docs.google.com/document/d/1TWwhFmji458pAycIzHSXn9rB8dsC8AZpyY7Qghsrwew/edit?usp=sharing)
