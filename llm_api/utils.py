from collections import Counter

import numpy as np

from functools import partial
from scipy.optimize import fmin

from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast
from datasets import Dataset


class OptimizeAUC:
    """
    Class for optimizing AUC by finding the best linear combination of model predictions.

    Attributes:
        coef_ (ndarray): The best coefficients found for the ensemble.
    """

    def __init__(self):
        self.coef_ = None

    def _negative_auc(self, coef, X, y):
        """
        Compute negative AUC for a given set of ensemble coefficients.

        Args:
            coef (ndarray): Coefficients for each model (same length as # models).
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).
            y (ndarray): True labels (shape: [n_samples]).

        Returns:
            float: Negative AUC score.
        """
        # Weighted predictions
        predictions = np.sum(X * coef, axis=1)
        auc_score = metrics.roc_auc_score(y, predictions)
        return -auc_score

    def fit(self, X, y):
        """
        Fit the ensemble coefficients to maximize AUC (minimize negative AUC).

        Args:
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).
            y (ndarray): True labels (shape: [n_samples]).
        """
        from numpy.random import dirichlet
        loss_partial = partial(self._negative_auc, X=X, y=y)

        # Initialize coefficients from a Dirichlet distribution (to sum to 1).
        initial_coef = dirichlet(np.ones(X.shape[1]), size=1)[0]

        # Minimize negative AUC
        self.coef_ = fmin(loss_partial, initial_coef, disp=True)

    def predict(self, X):
        """
        Generate ensemble predictions with the learned coefficients.

        Args:
            X (ndarray): Predictions from multiple models (shape: [n_samples, n_models]).

        Returns:
            ndarray: Final predictions (shape: [n_samples]).
        """
        return np.sum(X * self.coef_, axis=1)


def get_optimal_vocab_size(tokenized_corpus, threshold=0.999):
    """
    Given a tokenized corpus, find the vocabulary size that covers
    `threshold` fraction of the token occurrences.

    Args:
        tokenized_corpus (List[List[str]]): A list of tokenized texts.
        threshold (float): The fraction of total tokens you want to cover.

    Returns:
        int: The calculated vocabulary size.
    """
    all_tokens = [token for text in tokenized_corpus for token in text]
    word_freq = Counter(all_tokens)
    total_tokens = sum(word_freq.values())
    cumulative_freq = 0
    vocab_size = 0

    for token, freq in word_freq.most_common():
        cumulative_freq += freq / total_tokens
        vocab_size += 1
        if cumulative_freq >= threshold:
            break
    return vocab_size


def build_tokenizer(texts, vocab_size, lowercase=False):
    """
    Build and train a Byte-Pair Encoding tokenizer using Hugging Face's Tokenizer.

    Args:
        texts (List[str]): The list of texts to train the tokenizer on.
        vocab_size (int): Desired vocabulary size for BPE.
        lowercase (bool): Whether to lowercase the text before tokenization.

    Returns:
        PreTrainedTokenizerFast: The trained tokenizer.
    """
    # Create raw BPE tokenizer
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    if lowercase:
        raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    else:
        raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Build a HF Dataset from texts for streaming large datasets
    dataset = Dataset.from_dict({'text': texts})

    # Simple chunked iterator
    def train_corp_iter():
        batch_size = 1000
        for i in range(0, len(dataset), batch_size):
            yield dataset[i: i + batch_size]["text"]

    # Train from iterator
    raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

    # Wrap in a PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return tokenizer


def dummy(text):
    """A dummy tokenizer/preprocessor function to pass raw tokens to TfidfVectorizer."""
    return text


def get_ensemble_model():
    """
    Construct an ensemble model with VotingClassifier.

    Returns:
        VotingClassifier: An ensemble model composed of MNB, SGD, LGBM, CatBoost.
    """
    mnb = MultinomialNB(alpha=0.02)
    sgd = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")

    lgb_params = {
        'n_iter': 1500,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05073909898961407,
        'colsample_bytree': 0.726023996436955,
        'colsample_bynode': 0.5803681307354022,
        'lambda_l1': 8.562963348932286,
        'lambda_l2': 4.893256185259296,
        'min_data_in_leaf': 115,
        'max_depth': 23,
        'max_bin': 898
    }
    lgb_clf = LGBMClassifier(**lgb_params)

    cat_clf = CatBoostClassifier(
        iterations=1000,
        verbose=False,
        l2_leaf_reg=6.6591278779517808,
        learning_rate=0.005689066836106983 / 2,
        allow_const_label=True,
        loss_function='CrossEntropy',
        random_seed=1234
    )

    # Weights can be tuned
    weights = [0.07, 0.41, 0.41, 0.41]

    ensemble = VotingClassifier(
        estimators=[
            ('mnb', mnb),
            ('sgd', sgd),
            ('lgb', lgb_clf),
            ('cat', cat_clf)
        ],
        weights=weights,
        voting='soft',
        n_jobs=-1
    )
    return ensemble
