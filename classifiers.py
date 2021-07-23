import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


class Base:
    def __init__(self):
        pass

    def read_data(self, fname, lower_case=False, colnames=['truth', 'text']):
        df = pd.read_csv(fname, sep='\t', header=None, names=colnames)
        df['truth'] = df['truth'].str.replace('__label__', '')
        df['truth'] = df['truth'].astype(int).astype('category')
        if lower_case:
            df['text'] = df['text'].str.lower()
        return df

    def accuracy(self, df):
        acc = accuracy_score(df['truth'], df['pred'])*100
        f1 = f1_score(df['truth'], df['pred'], average='macro')*100
        print("Accuracy: {:.3f}\nMacro F1-score: {:.3f}".format(acc, f1))


class LogisticRegressionSentiment(Base):
    def __init__(self, model_file):
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(
                    solver='newton-cg',
                    multi_class='auto',
                )),
            ]
        )

    def predict(self, train_file, test_file, lower_case):
        # "Train model using sklearn pipeline"
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        # Fit the learner to the test data
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class SVMSentiment(Base):
    def __init__(self, model_file):
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    learning_rate='optimal',
                    tol=None,
                )),
            ]
        )

    def predict(self, train_file, test_file, lower_case):
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class DecisionTree(Base):
    def __init__(self, model_file: str = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', DecisionTreeClassifier(
                    class_weight='balanced', criterion='gini')),
            ]
        )

    def predict(self, train_file, test_file, lower_case):
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df


class NaivesBayes(Base):
    def __init__(self, model_file: str = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        self.pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ]
        )

    def predict(self, train_file, test_file, lower_case):
        train_df = self.read_data(train_file, lower_case)
        learner = self.pipeline.fit(train_df['text'], train_df['truth'])
        test_df = self.read_data(test_file, lower_case)
        test_df['pred'] = learner.predict(test_df['text'])
        return test_df
