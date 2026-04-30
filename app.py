import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load Models ───────────────────────────────────────────────────────────────
clf        = joblib.load('clf.pkl')
reg        = joblib.load('reg.pkl')
scaler     = joblib.load('scaler.pkl')
brand_meta = joblib.load('brand_meta.pkl')

features = [
    'Brand_score', 'Hydration_score', 'Medically_approved', 'Causes_irritation',
    'packaging_score', 'Fragrance', 'Sensitive', 'Oily', 'Normal', 'Price_per_ml',
    'oily_sensitive_risk', 'sensitive_irritation', 'oily_fragrance_risk',
    'Product_type_Cleanser', 'Product_type_Moisturizer',
    'Product_type_Serum', 'Product_type_Sunscreen'
]

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Skincare Market Predictor", page_icon="🧴", layout="centered")
st.title("🧴 Skincare Product Market Predictor")
st.markdown("Fill in the product details below to predict its market performance.")

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Product Details")

product_name = st.text_input("Product Name", placeholder="e.g. GlowBoost Serum")
product_type = st.selectbox("Product Type", ["Cleanser", "Moisturizer", "Serum", "Sunscreen"])
price_per_ml = st.number_input("Price per ml (₹)", min_value=0.0, step=0.5, value=5.0)

col1, col2 = st.columns(2)
with col1:
    packaging  = st.slider("Packaging Score", 1.0, 10.0, 5.0)
    hydration  = st.slider("Hydration Score", 1.0, 10.0, 5.0)
with col2:
    medically_appr = st.radio("Medically Approved", [1, 0], format_func=lambda x: "Yes" if x else "No")
    causes_irrit   = st.radio("Causes Irritation",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    fragrance      = st.radio("Contains Fragrance", [0, 1], format_func=lambda x: "Yes" if x else "No")

st.subheader("Skin Type Suitability")
col3, col4, col5 = st.columns(3)
with col3:
    sensitive = st.checkbox("Sensitive")
with col4:
    oily   = st.checkbox("Oily")
with col5:
    normal = st.checkbox("Normal")

st.subheader("Brand Details")
avg_rating_brand = st.slider("Brand Avg Rating on Other Products", 1.0, 5.0, 3.5)
years_in_market  = st.number_input("Brand Years in Market", min_value=0, max_value=100, value=5)

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Market Performance", use_container_width=True):

    # Brand score (same formula as notebook)
    rating_norm = (avg_rating_brand - brand_meta['rating_min']) / \
                  (brand_meta['rating_max'] - brand_meta['rating_min'])
    years_norm  = (years_in_market  - brand_meta['years_min']) / \
                  (brand_meta['years_max'] - brand_meta['years_min'])
    brand_score = (rating_norm * 0.75) + (years_norm * 0.25)

    input_features = pd.DataFrame([{
        'Brand_score'             : brand_score,
        'Hydration_score'         : hydration,
        'Medically_approved'      : int(medically_appr),
        'Causes_irritation'       : int(causes_irrit),
        'packaging_score'         : packaging,
        'Fragrance'               : int(fragrance),
        'Sensitive'               : int(sensitive),
        'Oily'                    : int(oily),
        'Normal'                  : int(normal),
        'Price_per_ml'            : price_per_ml,
        'oily_sensitive_risk'     : int(oily) * int(sensitive),
        'sensitive_irritation'    : int(sensitive) * int(causes_irrit),
        'oily_fragrance_risk'     : int(oily) * int(fragrance),
        'Product_type_Cleanser'   : 1 if product_type == 'Cleanser'    else 0,
        'Product_type_Moisturizer': 1 if product_type == 'Moisturizer' else 0,
        'Product_type_Serum'      : 1 if product_type == 'Serum'       else 0,
        'Product_type_Sunscreen'  : 1 if product_type == 'Sunscreen'   else 0,
    }])

    input_scaled    = scaler.transform(input_features[features])
    pred_rating     = round(reg.predict(input_scaled)[0], 2)
    pred_class      = clf.predict(input_scaled)[0]
    pred_confidence = round(max(clf.predict_proba(input_scaled)[0]) * 100, 1)

    # Verdict logic
    def market_verdict(pred_class, rating):
        if   pred_class == "High"   and rating >= 4.2: return "🏆 TOP RANKER",           "Highly popular AND top rated. This product is a market leader."
        elif pred_class == "High"   and rating >= 3.8: return "👥 CROWD FAVOURITE",      "Strong reach but average ratings — good launch, needs quality improvement."
        elif pred_class == "High":                     return "📉 OVERHYPED – DECLINING", "Will get attention but poor ratings will drive a fast decline."
        elif pred_class == "Medium" and rating >= 4.2: return "🚀 RISING STAR",          "Excellent quality with moderate reach. High growth potential."
        elif pred_class == "Medium":                   return "✅ SOLID PERFORMER",       "Steady market presence with decent ratings. Reliable, not spectacular."
        elif rating >= 4.2:                            return "💎 HIDDEN GEM",            "Quality is there but reach will be limited — needs strong marketing."
        else:                                          return "❌ POOR PRODUCT",          "Low popularity and low ratings predicted. Revisit formulation."

    verdict_label, verdict_desc = market_verdict(pred_class, pred_rating)

    # ── Results Display ───────────────────────────────────────────────────────
    st.divider()
    st.subheader(f"Report — {product_name or 'My Product'}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Avg Rating", f"{pred_rating} / 5.0")
    with col_b:
        st.metric("Market Class", f"{pred_class} ({pred_confidence}% confidence)")

    color = {"High": "green", "Medium": "orange", "Low": "red"}[pred_class]
    st.markdown(f"### Verdict: {verdict_label}")
    st.info(verdict_desc)