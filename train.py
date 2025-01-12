import gc
import yaml
import numpy as np
import pandas as pd

from scipy.sparse import vstack

from llm_api.settings import config, LOGGER
from llm_api.utils import OptimizeAUC, get_optimal_vocab_size, build_tokenizer, dummy, get_ensemble_model

# SKLearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Datasets
from tqdm.auto import tqdm


train_data_path = config["DATA_PATHS"]["TRAIN_DATA"]
test_data_path = config["DATA_PATHS"]["TEST_DATA"]
submission_data_path = config["DATA_PATHS"]["SUBMISSION_DATA"]
sep = config["PARAMS"]["SEPARATOR"]
random_seed = config["PARAMS"]["RANDOM_SEED"]




def main():
    """
    Orchestrates the data reading, tokenization, TF-IDF vectorization,
    adversarial filtering, pseudo-labeling, and final ensemble training.
    """
    LOGGER.info("Reading data...")
    # Adjust file paths as needed
    test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
    sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
    train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

    # Basic cleaning
    train = train.drop_duplicates(subset=['text']).reset_index(drop=True)
    y_train = train['label'].values

    LOGGER.info("Building tokenizer...")
    # We combine texts from both train and test for tokenizer training if needed
    # Or you might want to restrict to train only, depending on your scenario
    combined_texts = train['text'].tolist() + test['text'].tolist()

    # For demonstration, we fix vocab size to 30522
    tokenizer = build_tokenizer(texts=combined_texts, vocab_size=30522, lowercase=False)

    LOGGER.info("Tokenizing data...")
    tokenized_train = [tokenizer.tokenize(t) for t in tqdm(train['text'].tolist())]
    tokenized_test = [tokenizer.tokenize(t) for t in tqdm(test['text'].tolist())]

    # Optional: get optimal vocab size
    # optimal_vocab_size = get_optimal_vocab_size(tokenized_train, threshold=0.999)
    # logger.info(f"Optimal vocab size: {optimal_vocab_size}")

    LOGGER.info("Vectorizing with TF-IDF...")
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

    LOGGER.info("Adversarial filtering...")
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

    del X_adv, y_adv, X_train_adv, X_val_adv, y_train_adv, y_val_adv, train_probs
    gc.collect()

    LOGGER.info("Pseudo-labeling high-confidence test samples...")
    pseudo_model = get_ensemble_model()
    pseudo_model.fit(tf_train_filtered, y_train_filtered)

    test_preds = pseudo_model.predict_proba(tf_test)[:, 1]
    high_confidence_mask = (test_preds > 0.9) | (test_preds < 0.1)
    pseudo_labels = (test_preds > 0.5).astype(int)

    tf_train_enhanced = vstack((tf_train_filtered, tf_test[high_confidence_mask]))
    y_train_enhanced = np.concatenate((y_train_filtered, pseudo_labels[high_confidence_mask]))

    del pseudo_model, tf_train_filtered, y_train_filtered, pseudo_labels
    gc.collect()

    LOGGER.info("Final model training with cross-validation...")
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
        LOGGER.info(f"Fold {fold} AUC: {fold_auc:.4f}")

        test_preds_matrix[:, fold] = ensemble_model.predict_proba(tf_test)[:, 1]

    LOGGER.info("Optimizing ensemble weights with AUC...")
    opt_auc = OptimizeAUC()
    opt_auc.fit(oof_preds, y_train_enhanced)
    W = opt_auc.coef_

    combined_test_preds = np.sum(test_preds_matrix * W, axis=1)
    test['generated'] = combined_test_preds

    submission = pd.DataFrame({
        'id': test["id"],
        'generated': test['generated']
    })
    submission.to_csv('submission.csv', index=False)
    LOGGER.info("Submission file saved as submission.csv.")


if __name__ == "__main__":
    main()
