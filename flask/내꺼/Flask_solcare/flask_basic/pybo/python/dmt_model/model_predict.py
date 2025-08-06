import joblib

def load_model(path="python/model_dmt/model_v5.pkl"):
    return joblib.load(path)

def predict(df, model):
    pred = model.predict(df)[0]
    proba = float(model.predict_proba(df)[0][1])
    return pred, proba

def get_feature_importance(model):
    return [
        {"feature": name, "importance": float(score)}
        for name, score in zip(model.get_booster().feature_names, model.feature_importances_)
    ]



