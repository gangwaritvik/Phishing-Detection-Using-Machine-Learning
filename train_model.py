import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from feature_extractor import get_feature_dict

PHIUSIIL_FEATURE_COLS = [
    'URLLength','DomainLength','IsDomainIP','URLSimilarityIndex',
    'CharContinuationRate','TLDLegitimateProb','URLCharProb','TLDLength',
    'NoOfSubDomain','HasObfuscation','NoOfObfuscatedChar','ObfuscationRatio',
    'NoOfLettersInURL','LetterRatioInURL','NoOfDegitsInURL','DegitRatioInURL',
    'NoOfEqualsInURL','NoOfQMarkInURL','NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL','SpacialCharRatioInURL','IsHTTPS',
    'LineOfCode','LargestLineLength','HasTitle','DomainTitleMatchScore',
    'URLTitleMatchScore','HasFavicon','Robots','IsResponsive',
    'NoOfURLRedirect','NoOfSelfRedirect','HasDescription','NoOfPopup',
    'NoOfiFrame','HasExternalFormSubmit','HasSocialNet','HasSubmitButton',
    'HasHiddenFields','HasPasswordField','Bank','Pay','Crypto',
    'HasCopyrightInfo','NoOfImage','NoOfCSS','NoOfJS','NoOfSelfRef',
    'NoOfEmptyRef','NoOfExternalRef',
]

def load_phiusiil(csv_path):
    df = pd.read_csv(csv_path)

    label_col = None
    for candidate in ['label','Label','phishing','class','target']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = df.columns[-1]

    y = df[label_col]

    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    feature_cols = [c for c in PHIUSIIL_FEATURE_COLS if c in df.columns]

    X = df.reindex(columns=feature_cols, fill_value=0)

    return X.values, np.array(y), feature_cols


def train(csv_path, output='model.pkl'):

    X, y, feature_names = load_phiusiil(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    print("Using XGBoost")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print("5-fold CV AUC:", cv_scores.mean())

    with open(output, 'wb') as f:
        pickle.dump({'model': clf, 'feature_names': feature_names}, f)

    print("Model saved to", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', default='model.pkl')

    args = parser.parse_args()

    train(args.dataset, args.output)