import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Try to import XGBoost if available
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

st.set_page_config(page_title="Auto EDA + Model Playground", layout="wide")

st.title("📊 Auto EDA + Model Playground (Streamlit)")
st.markdown(
    """
    Upload a dataset (CSV or Excel) or provide a sample path. Select the target column, choose features (or use all),
    pick a model, and adjust the train/test split. The app will run EDA and train the model, showing metrics and plots.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])    
    use_sample = st.checkbox("Use sample dataset (Car Price sample)", value=False)
    if not uploaded_file and not use_sample:
        st.info("Upload a CSV/Excel or check 'Use sample dataset' to try the app.")

    random_state = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Test set proportion", 0.05, 0.5, 0.2, step=0.05)
    scale_features = st.checkbox("Scale features (StandardScaler)", value=False)
    max_rows_pairplot = st.number_input("Max rows for pairplot", min_value=200, max_value=2000, value=500, step=100)

    st.markdown("---")
    st.subheader("Model selection")
    model_choice = st.selectbox(
        "Choose a model",
        options=[
            "Linear Regression",
            "Ridge",
            "Lasso",
            "Random Forest",
            "Gradient Boosting"
        ] + (["XGBoost"] if HAS_XGB else [])
    )

    st.markdown("**Hyperparameters (quick tuning)**")
    if model_choice in ["Random Forest"]:
        n_estimators = st.slider("n_estimators", 10, 1000, 100, step=10)
        max_depth = st.slider("max_depth (0=None)", 0, 50, 10)
        if max_depth == 0:
            max_depth = None
    elif model_choice in ["Gradient Boosting", "XGBoost"]:
        n_estimators = st.slider("n_estimators", 10, 1000, 100, step=10)
        learning_rate = st.number_input("learning_rate", 0.001, 1.0, 0.1)
        max_depth = st.slider("max_depth (0=None)", 0, 50, 6)
        if max_depth == 0:
            max_depth = None
    else:
        n_estimators = 100
        learning_rate = 0.1
        max_depth = None

# Load data
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

@st.cache_data
def load_sample_data():
    n = 1000
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        'year': rng.randint(2000, 2021, size=n),
        'mileage': rng.randint(5000, 200000, size=n),
        'engine_size': rng.uniform(0.8, 4.5, size=n).round(2),
        'power': rng.randint(50, 400, size=n),
        'price': np.abs((2025 - rng.randint(2000, 2021, size=n))*100 + rng.normal(0, 2000, size=n)).round(0)
    })
    return df

if use_sample and not uploaded_file:
    df = load_sample_data()
elif uploaded_file:
    try:
        df = load_data(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()
else:
    df = None


if df is not None:
    st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    st.header("🔎 Exploratory Data Analysis")
    row1_col1, row1_col2 = st.columns([1, 2])
    with row1_col1:
        st.subheader("Preview")
        st.dataframe(df.head(10))
        st.write("**Shape:**", df.shape)

        # select target and features
        all_columns = df.columns.tolist()
        target_col = st.selectbox("Select the target column", options=all_columns, index=len(all_columns)-1)
        default_features = [c for c in all_columns if c != target_col]
        feature_selector = st.multiselect("Select features (leave empty to use all except target)", options=default_features, default=default_features)

    with row1_col2:
        st.subheader("Summary statistics")
        st.write(df.describe(include='all'))

    # Missing values and dtypes
    st.subheader("Data types & Missing values")
    miss = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'missing_pct': (df.isna().mean()*100).round(2)})
    st.dataframe(miss)

    # Visual EDA
    st.subheader("Visualizations")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.write("Histogram / Distribution of numeric features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            sel_num = st.multiselect("Pick numeric columns to show histograms", options=numeric_cols, default=numeric_cols[:3])
            for c in sel_num:
                fig, ax = plt.subplots()
                sns.histplot(df[c].dropna(), kde=True)
                ax.set_title(f"Distribution: {c}")
                st.pyplot(fig)
    with viz_col2:
        if numeric_cols:
            st.write("Correlation matrix (numeric)")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', ax=ax)
            st.pyplot(fig)

    # Pairplot (only if not too large)
    if df.shape[0] <= max_rows_pairplot and len(numeric_cols) <= 6:
        st.subheader("Pairplot (small datasets)")
        try:
            fig = sns.pairplot(df[numeric_cols]).fig
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not draw pairplot: {e}")

    # Prepare data for modeling
    X = df[feature_selector].copy()
    y = df[target_col].copy()

    # basic preprocessing: drop rows with missing target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # numeric-only simple approach: drop non-numeric features or one-hot encode small cardinality
    X_proc = X.copy()
    cat_cols = X_proc.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        st.info(f"Found categorical columns: {cat_cols}. They will be one-hot encoded (careful: may increase columns).")
        X_proc = pd.get_dummies(X_proc, columns=cat_cols, drop_first=True)

    # Impute simple missing values
    X_proc = X_proc.fillna(X_proc.median(numeric_only=True))

    if scale_features:
        scaler = StandardScaler()
        X_proc = pd.DataFrame(scaler.fit_transform(X_proc), columns=X_proc.columns)

    st.markdown("---")
    st.header("⚙️ Modeling")

    # Split
    test_size = float(test_size)
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=int(random_state))
    st.write(f"Train size: {X_train.shape[0]} rows, Test size: {X_test.shape[0]} rows")

    # Build model based on selection
    def build_model(name: str):
        if name == 'Linear Regression':
            return LinearRegression()
        if name == 'Ridge':
            return Ridge()
        if name == 'Lasso':
            return Lasso()
        if name == 'Random Forest':
            return RandomForestRegressor(n_estimators=int(n_estimators), max_depth=max_depth, random_state=int(random_state))
        if name == 'Gradient Boosting':
            return GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=max_depth, random_state=int(random_state))
        if name == 'XGBoost' and HAS_XGB:
            return XGBRegressor(n_estimators=int(n_estimators), learning_rate=float(learning_rate), max_depth=max_depth, random_state=int(random_state), verbosity=0)
        raise ValueError('Unsupported model')

    model = build_model(model_choice)

    # Train button
    if st.button("Train model"):
        t0 = time.time()
        with st.spinner("Training..."):
            model.fit(X_train, y_train)
        t1 = time.time()
        st.success(f"Training completed in {t1-t0:.2f}s")

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Metrics
        def regression_metrics(y_true, y_pred):
            return {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred)
            }

        st.subheader("Metrics")
        cols = st.columns(2)
        with cols[0]:
            st.write("**Train metrics**")
            st.json(regression_metrics(y_train, y_pred_train))
        with cols[1]:
            st.write("**Test metrics**")
            st.json(regression_metrics(y_test, y_pred_test))

        # Feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                fi = pd.Series(model.feature_importances_, index=X_proc.columns).sort_values(ascending=False)
                st.subheader("Feature importances")
                st.bar_chart(fi.head(20))
            elif hasattr(model, 'coef_'):
                coef = pd.Series(model.coef_, index=X_proc.columns).sort_values(key=lambda s: np.abs(s), ascending=False)
                st.subheader("Coefficients (by absolute value)")
                st.write(coef.head(20))
        except Exception as e:
            st.warning(f"Could not extract feature importances: {e}")

        # Prediction vs Actual plot
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_test, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted (Test set)')
        st.pyplot(fig)

        # Show a small table of predictions
        st.subheader("Some predictions (test sample)")
        sample_out = X_test.copy()
        sample_out['actual'] = y_test
        sample_out['predicted'] = y_pred_test
        st.dataframe(sample_out.sort_index().head(15))

    st.markdown("---")
    st.info("Tip: For better performance on categorical-heavy data consider targeted preprocessing (Label Encoding, Target Encoding) and for time-series problems use dedicated pipelines.")

else:
    st.stop()

