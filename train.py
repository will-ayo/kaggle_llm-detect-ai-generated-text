import gc
import numpy as np
import pandas as pd

from collections import Counter
from scipy.sparse import vstack
from llm_api.utils import OptimizeAUC


# SKLearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

# LightGBM, CatBoost
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Tokenizers / Transformers
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

# Datasets
from datasets import Dataset
from tqdm.auto import tqdm


import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
        verbose=0,
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


def main():
    """
    Orchestrates the data reading, tokenization, TF-IDF vectorization,
    adversarial filtering, pseudo-labeling, and final ensemble training.
    """
    logger.info("Reading data...")
    # Adjust file paths as needed
    test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
    sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
    train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

    # Basic cleaning
    train = train.drop_duplicates(subset=['text']).reset_index(drop=True)
    y_train = train['label'].values

    logger.info("Building tokenizer...")
    # We combine texts from both train and test for tokenizer training if needed
    # Or you might want to restrict to train only, depending on your scenario
    combined_texts = train['text'].tolist() + test['text'].tolist()

    # For demonstration, we fix vocab size to 30522
    tokenizer = build_tokenizer(texts=combined_texts, vocab_size=30522, lowercase=False)

    logger.info("Tokenizing data...")
    tokenized_train = [tokenizer.tokenize(t) for t in tqdm(train['text'].tolist())]
    tokenized_test = [tokenizer.tokenize(t) for t in tqdm(test['text'].tolist())]

    # Optional: get optimal vocab size
    # optimal_vocab_size = get_optimal_vocab_size(tokenized_train, threshold=0.999)
    # logger.info(f"Optimal vocab size: {optimal_vocab_size}")

    logger.info("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents='unicode'
    )
    # Fit only on the train set or combined set depending on your preference
    vectorizer.fit(tokenized_train + tokenized_test)

    # Retrieve vocab if needed
    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(
        ngram_range=(3, 5),
        lowercase=False,
        sublinear_tf=True,
        vocabulary=vocab,
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
        token_pattern=None,
        strip_accents='unicode'
    )

    tf_train = vectorizer.fit_transform(tokenized_train)
    tf_test = vectorizer.transform(tokenized_test)

    # Clean up large variables
    del tokenized_train, tokenized_test, vectorizer, combined_texts
    gc.collect()

    logger.info("Adversarial filtering...")
    # Combine train+test into a single dataset
    X_adv = vstack((tf_train, tf_test))
    y_adv = np.array([0] * tf_train.shape[0] + [1] * tf_test.shape[0])

    # Train a simple adversarial model
    X_train_adv, X_val_adv, y_train_adv, y_val_adv = train_test_split(
        X_adv, y_adv, test_size=0.3, random_state=42
    )
    adv_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    adv_model.fit(X_train_adv, y_train_adv)

    # Filter out suspicious train samples
    train_probs = adv_model.predict_proba(tf_train)[:, 1]
    threshold = 0.15
    selected_train_indices = train_probs > threshold

    # If no variety in predictions, skip filtering
    if len(np.unique(train_probs)) != 1:
        tf_train_filtered = tf_train[selected_train_indices]
        y_train_filtered = y_train[selected_train_indices]
    else:
        tf_train_filtered = tf_train
        y_train_filtered = y_train

    # Clean up
    del X_adv, y_adv, X_train_adv, X_val_adv, y_train_adv, y_val_adv, train_probs
    gc.collect()

    logger.info("Pseudo-labeling high-confidence test samples...")
    pseudo_model = get_ensemble_model()
    pseudo_model.fit(tf_train_filtered, y_train_filtered)

    test_preds = pseudo_model.predict_proba(tf_test)[:, 1]
    high_confidence_mask = (test_preds > 0.9) | (test_preds < 0.1)
    pseudo_labels = (test_preds > 0.5).astype(int)

    tf_train_enhanced = vstack((tf_train_filtered, tf_test[high_confidence_mask]))
    y_train_enhanced = np.concatenate((y_train_filtered, pseudo_labels[high_confidence_mask]))

    # Cleanup
    del pseudo_model, tf_train_filtered, y_train_filtered, pseudo_labels
    gc.collect()

    logger.info("Final model training with cross-validation...")
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros((tf_train_enhanced.shape[0], n_splits))
    test_preds_matrix = np.zeros((tf_test.shape[0], n_splits))

    for fold, (trn_idx, val_idx) in enumerate(skf.split(tf_train_enhanced, y_train_enhanced)):
        X_trn, X_val = tf_train_enhanced[trn_idx], tf_train_enhanced[val_idx]
        y_trn, y_val = y_train_enhanced[trn_idx], y_train_enhanced[val_idx]

        ensemble_model = get_ensemble_model()
        ensemble_model.fit(X_trn, y_trn)

        val_preds = ensemble_model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx, fold] = val_preds
        fold_auc = roc_auc_score(y_val, val_preds)
        logger.info(f"Fold {fold} AUC: {fold_auc:.4f}")

        test_preds_matrix[:, fold] = ensemble_model.predict_proba(tf_test)[:, 1]

    # Optimize AUC with linear combination
    logger.info("Optimizing ensemble weights with AUC...")
    opt_auc = OptimizeAUC()
    opt_auc.fit(oof_preds, y_train_enhanced)
    W = opt_auc.coef_

    # Final predictions
    combined_test_preds = np.sum(test_preds_matrix * W, axis=1)
    test['generated'] = combined_test_preds

    # Save submission
    submission = pd.DataFrame({
        'id': test["id"],
        'generated': test['generated']
    })
    submission.to_csv('submission.csv', index=False)
    logger.info("Submission file saved as submission.csv.")


if __name__ == "__main__":
    main()