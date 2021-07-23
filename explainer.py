import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.pipeline
from pathlib import Path
from typing import List, Any
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
import spacy
import pandas as pd
import tkinter as tk
from tkinter import Tk, simpledialog
from tkinter.ttk import Button, Entry

ROOT = tk.Tk()
ROOT.withdraw()

METHODS = {
    'logistic': {
        'class': "LogisticExplainer",
        'file': "sst_train.txt"
    },
    'svm': {
        'class': "SVMExplainer",
        'file': "sst_train.txt"
    },
    'bayes': {
        'class': "NaivesBayes",
        'file': "sst_train.txt"
    },
    'decision': {
        'class': "DecisionTree",
        'file': "sst_train.txt"
    }
}


def tokenizer(text: str):
    nlp = spacy.blank('en')
    nlp.create_pipe('sentencizer')
    doc = nlp(text)
    tokenized_text = ' '.join(token.text for token in doc)
    return tokenized_text


def explainer_class(method: str, filename: str):
    classname = METHODS[method]['class']
    class_ = globals()[classname]
    return class_(filename)


def results(results=[]):
    sizes = []
    explode = ()
    labels = ["Strongly Negative", "Weakly Negative",
              "Neutral", "Weakly Positive", "Strongly Positive", ]
    for i in range(len(results)):
        print(f"Value of {labels[i]} : {results[i]}")
        sizes.append(results[i]*500)
    final_label = max(results)
    index_max = np.argmax(results)
    print()
    print("The resultant emotion is ",
          labels[index_max], "as it has the highest predicted value of: ", final_label)
    colors = ['gold', 'yellowgreen', 'lightcoral',
              'lightskyblue', 'red']
    explode = [0.1 if i == index_max else 0 for i in range(5)]
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.show()


class LogisticExplainer:

    def __init__(self, path_to_train_data: str):
        # "Input training data path for training Logistic Regression classifier"
        import pandas as pd
        # Read in training data set
        self.train_df = pd.read_csv(
            path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace(
            '__label__', '')
        # Categorical data type for truth labels
        self.train_df['truth'] = self.train_df['truth'].astype(
            int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(solver='newton-cg', multi_class='auto')),
            ]
        )
        classifier = pipeline.fit(
            self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        classifier = self.train()
        probs = classifier.predict_proba(texts)
        results(probs[0])
        return probs


class DecisionTree:

    def __init__(self, path_to_train_data: str) -> None:
        import pandas as pd
        self.train_df = pd.read_csv(
            path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace(
            '__label__', '')
        self.train_df['truth'] = self.train_df['truth'].astype(
            int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', DecisionTreeClassifier(
                    class_weight='balanced', criterion='gini')),
            ]
        )
        classifier = pipeline.fit(
            self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):

        classifier = self.train()
        probs = classifier.predict_proba(texts)
        results(probs[0])
        return probs


class SVMExplainer:

    def __init__(self, path_to_train_data: str) -> None:
        import pandas as pd
        self.train_df = pd.read_csv(
            path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace(
            '__label__', '')
        self.train_df['truth'] = self.train_df['truth'].astype(
            int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='modified_huber',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    tol=None,
                )),
            ]
        )
        classifier = pipeline.fit(
            self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):
        classifier = self.train()
        probs = classifier.predict_proba(texts)
        results(probs[0])
        return probs


class NaivesBayes:

    def __init__(self, path_to_train_data: str) -> None:
        import pandas as pd
        # Read in training data set
        self.train_df = pd.read_csv(
            path_to_train_data, sep='\t', header=None, names=["truth", "text"])
        self.train_df['truth'] = self.train_df['truth'].str.replace(
            '__label__', '')
        self.train_df['truth'] = self.train_df['truth'].astype(
            int).astype('category')

    def train(self) -> sklearn.pipeline.Pipeline:
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
            ]
        )
        classifier = pipeline.fit(
            self.train_df['text'], self.train_df['truth'])
        return classifier

    def predict(self, texts: List[str]) -> np.array([float, ...]):

        classifier = self.train()
        probs = classifier.predict_proba(texts)
        results(probs[0])
        return probs


def explainer(method: str,
              path_to_file: str,
              text: str,
              num_samples: int):

    model = explainer_class(method, path_to_file)
    predictor = model.predict
    explainer = LimeTextExplainer(
        split_expression=lambda x: x.split(),
        bow=False,
        class_names=[1, 2, 3, 4, 5]
    )
    print(text)
    exp = explainer.explain_instance(
        text,
        classifier_fn=predictor,
        top_labels=1,
        num_features=20,
        num_samples=num_samples,
    )


def main(samples: List[str], meth):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, help="Number of samples for explainer \
                        instance", default=2000)
    args = parser.parse_args()
    path_to_file = 'sst_train.txt'
    print("Method: {}".format(meth.upper()))
    for i, text in enumerate(samples):
        text = tokenizer(text)
        exp = explainer(meth, path_to_file, text, args.num_samples)


samples = [

    simpledialog.askstring(
        title="What's your statement?:",
        prompt="Type a statement for which you would like the sentiment to be evaluated")
]
main(samples, simpledialog.askstring(title="What's your method?:",
                                     prompt="Enter one of the following:\n -> svm - Support Vector Machine \n -> logistic - Logistic Regression \n -> decision - Decision Tree\n -> bayes - Naives Bayes"))
