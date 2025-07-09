"""
Train XGBoost classifier on generated data.
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def train(self, csv_path: str):
        df = pd.read_csv(csv_path)
        X = df.drop('label', axis=1)
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'grow_policy': 'lossguide'
        }
        model = xgb.train(params, dtrain, num_boost_round=200)
        preds = (model.predict(dtest) > 0.5).astype(int)
        print("Accuracy:", accuracy_score(y_test, preds))
        return model

    def save(self, model, path: str):
        model.save_model(path)
