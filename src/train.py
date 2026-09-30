
"""
Train the Shopzilla CSAT ANN.
Usage:
    python train.py --data data/eCommerce_Customer_support_data.csv
"""
import argparse, json, os, random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, classification_report
from features import build_features
from model import CSATNet

SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def main(args):
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(args.data)
    X = build_features(df)
    y = df["CSAT Score"].astype(int).to_numpy() - 1

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=SEED
    )

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]), cat_cols)
    ], sparse_threshold=1.0)

    A = preprocessor.fit_transform(X_train)
    B = preprocessor.transform(X_val)
    C = preprocessor.transform(X_test)

    svd = TruncatedSVD(n_components=128, random_state=SEED)
    A = svd.fit_transform(A).astype("float32")
    B = svd.transform(B).astype("float32")
    C = svd.transform(C).astype("float32")

    Xtr=torch.tensor(A); Xv=torch.tensor(B); Xte=torch.tensor(C)
    ytr=torch.tensor(y_train, dtype=torch.long)
    yv=torch.tensor(y_val, dtype=torch.long)
    yte=torch.tensor(y_test, dtype=torch.long)

    counts=np.bincount(y_train, minlength=5)
    weights=(len(y_train)/(5*counts))**0.25
    weights=weights/weights.mean()
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))

    model=CSATNet(128,5)
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader=DataLoader(TensorDataset(Xtr,ytr), batch_size=512, shuffle=True)

    best_state=None; best_macro=-1
    for epoch in range(args.epochs):
        model.train()
        for xb,yb in loader:
            optimizer.zero_grad()
            loss=criterion(model(xb),yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pv=model(Xv).argmax(1).numpy()
        macro=f1_score(y_val,pv,average="macro")
        if macro > best_macro:
            best_macro=macro
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {epoch+1}: validation accuracy={accuracy_score(y_val,pv):.4f}, macro-F1={macro:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred=model(Xte).argmax(1).numpy()

    metrics={
        "accuracy": float(accuracy_score(y_test,pred)),
        "macro_f1": float(f1_score(y_test,pred,average="macro")),
        "weighted_f1": float(f1_score(y_test,pred,average="weighted")),
        "baseline_accuracy": float(np.bincount(y).max()/len(y)),
        "svd_explained_variance": float(svd.explained_variance_ratio_.sum())
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test,pred,target_names=["1","2","3","4","5"],zero_division=0))

    torch.save(model.state_dict(),"models/model_state_dict.pt")
    joblib.dump(preprocessor,"models/preprocessor.joblib")
    joblib.dump(svd,"models/svd.joblib")
    with open("models/metrics.json","w") as f: json.dump(metrics,f,indent=2)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="data/eCommerce_Customer_support_data.csv")
    p.add_argument("--epochs",type=int,default=8)
    main(p.parse_args())
