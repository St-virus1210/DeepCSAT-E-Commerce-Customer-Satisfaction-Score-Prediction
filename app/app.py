
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
import joblib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from features import build_features
from model import CSATNet

st.set_page_config(page_title="Shopzilla CSAT Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_assets():
    pre=joblib.load(ROOT/"models/preprocessor.joblib")
    svd=joblib.load(ROOT/"models/svd.joblib")
    model=CSATNet(128,5)
    model.load_state_dict(torch.load(ROOT/"models/model_state_dict.pt", map_location="cpu"))
    model.eval()
    metrics=json.loads((ROOT/"models/metrics.json").read_text())
    return pre,svd,model,metrics

pre,svd,model,metrics=load_assets()

st.title("Shopzilla Customer Satisfaction Predictor")
st.caption("Deep Learning ANN demonstration for CSAT 1–5 classification")

with st.sidebar:
    st.header("Model")
    st.metric("Test Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("Macro-F1", f"{metrics['macro_f1']:.3f}")
    st.metric("Weighted-F1", f"{metrics['weighted_f1']:.3f}")
    st.info("The dataset is strongly imbalanced, so Macro-F1 should be considered alongside accuracy.")

st.subheader("Enter interaction details")
c1,c2,c3=st.columns(3)
with c1:
    channel=st.selectbox("Channel",["Inbound","Outcall","Email"])
    category=st.text_input("Category","Returns")
    subcategory=st.text_input("Sub-category","Return request")
    city=st.text_input("Customer City","")
with c2:
    product=st.text_input("Product Category","")
    price=st.number_input("Item Price",min_value=0.0,value=1000.0)
    handling=st.number_input("Connected Handling Time",min_value=0.0,value=0.0)
    tenure=st.selectbox("Tenure Bucket",[">90","On Job Training","31-60","0-30","61-90"])
with c3:
    shift=st.selectbox("Agent Shift",["Morning","Evening","Afternoon","Split","Night"])
    agent=st.text_input("Agent Name","")
    supervisor=st.text_input("Supervisor","")
    manager=st.text_input("Manager","")

issue_dt=st.text_input("Issue reported at","01/08/2023 11:13")
responded_dt=st.text_input("Issue responded","01/08/2023 11:47")
order_dt=st.text_input("Order date time","")
remarks=st.text_area("Customer Remarks","")

if st.button("Predict CSAT", type="primary"):
    row=pd.DataFrame([{
        "channel_name":channel,"category":category,"Sub-category":subcategory,
        "Customer Remarks":remarks,"Order_id":None,"order_date_time":order_dt,
        "Issue_reported at":issue_dt,"issue_responded":responded_dt,
        "Survey_response_Date":None,"Customer_City":city or None,
        "Product_category":product or None,"Item_price":price,
        "connected_handling_time":handling,"Agent_name":agent or None,
        "Supervisor":supervisor or None,"Manager":manager or None,
        "Tenure Bucket":tenure,"Agent Shift":shift
    }])
    feat=build_features(row)
    z=pre.transform(feat)
    z=svd.transform(z).astype("float32")
    with torch.no_grad():
        logits=model(torch.tensor(z))
        prob=torch.softmax(logits,dim=1).numpy()[0]
    pred=int(prob.argmax()+1)
    st.success(f"Predicted CSAT Score: {pred}")
    cols=st.columns(5)
    for i,p in enumerate(prob,1):
        cols[i-1].metric(f"CSAT {i}",f"{p:.1%}")
    st.progress(float(prob[pred-1]))
