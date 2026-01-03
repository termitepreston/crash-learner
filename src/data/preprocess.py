import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs preliminary cleaning: standardizing text, fixing typos,
    handling impossible values, and dropping high-missing columns.
    """
    df = df.copy()

    # 1. Standardize Target Variable
    acc_type_map = {
        "property damage only": "PDO",
        "Pdo": "PDO",
        "P.D.O": "PDO",
        "pdo": "PDO",
        "POD": "PDO",
        "Minor": "Minor",
        "Serious": "Serious",
        "Fatal": "Fatal",
    }
    df["Accident Type"] = (
        df["Accident Type"].map(acc_type_map).fillna(df["Accident Type"])
    )

    # 2. Fix Typographical Errors in Categorical Columns
    df["Wor"] = df["Wor"].replace(
        {
            "Augest": "August",
            "Pagume": "September",
            "Phagume": "September",
            "Wensday": "Wednesday",
        }
    )
    df["Day of the week"] = df["Day of the week"].replace({"Wensday": "Wednesday"})
    df["Veh Ownership"] = df["Veh Ownership"].replace({"Privategg": "Private"})

    # 3. Handle Numerical Outliers / Impossible Values
    # Amet (Year): 2000-2026
    df["Amet"] = pd.to_numeric(df["Amet"], errors="coerce")
    df.loc[(df["Amet"] < 2000) | (df["Amet"] > 2026), "Amet"] = np.nan

    # Age: 16-90
    df["Driver age"] = pd.to_numeric(df["Driver age"], errors="coerce")
    df.loc[(df["Driver age"] < 16) | (df["Driver age"] > 90), "Driver age"] = np.nan

    # Experience: < 60
    df["Driver experiance(years)"] = pd.to_numeric(
        df["Driver experiance(years)"], errors="coerce"
    )
    df.loc[df["Driver experiance(years)"] > 60, "Driver experiance(years)"] = np.nan

    # 4. Standardize 'Unknown' and textual inconsistencies
    # Driver Sex
    def clean_sex(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower().strip()
        if x.startswith("m"):
            return "Male"
        if x.startswith("f"):
            return "Female"
        return np.nan

    df["Driver Sex"] = df["Driver Sex"].apply(clean_sex)

    # Education
    df["Driver Education Level"] = df["Driver Education Level"].replace(
        "Unknown", np.nan
    )

    # Vehicle Defects
    def clean_defects(x):
        if pd.isna(x):
            return np.nan
        x = str(x).lower()
        if x.startswith("y"):
            return "Yes"
        if x.startswith("n"):
            return "No"
        return np.nan

    df["Vehicle Defects"] = df["Vehicle Defects"].apply(clean_defects)

    # 5. Drop Columns with > 70% Missing (identified in EDA)
    # Plus identifiers like 'Accident ID' which are not features

    drop_cols = [
        "Exiting/entering",
        "Region",
        "Driving License",
        "Contributory Action",
        "Zone",
        "Victim-3 Movement",
        "Victim-2 Movement",
        "Victim-3 Type",
        "Victim-3 Injury Severity",
        "Victim-3 Sex",
        "Victim-2 Sex",
        "Victim-2 Type",
        "Victim-2 Injury Severity",
        "Victim-1 Movement",
    ]
    # Filter to only drop columns that actually exist
    existing_drops = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drops)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features suitable for Neural Networks.
    """
    df = df.copy()

    # 1. Time Encoding (Cyclical)
    # Extract Hour from "15:00-16:00"
    def extract_hour(t):
        if pd.isna(t):
            return np.nan
        try:
            return int(str(t).split("-")[0].split(":")[0])
        except ValueError:
            return np.nan

    df["Hour"] = df["Time"].apply(extract_hour)

    # Handle missing hours (impute with mode or median before encoding)
    # Here we fill with median for simplicity before calculation
    hour_median = df["Hour"].median()
    df["Hour"] = df["Hour"].fillna(hour_median)

    # Create Sin/Cos features
    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Drop original Time and Hour columns
    df = df.drop(columns=["Time", "Hour"])

    return df


def build_preprocessor(X_train):
    """
    Constructs a Scikit-Learn ColumnTransformer for Imputation, Scaling, and Encoding.
    """
    # Identify column types
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Numeric Pipeline: Impute Median -> Scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical Pipeline: Impute Mode -> OneHot Encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_processed_data(df, target_col="Accident Type", test_size=0.15, val_size=0.15):
    """
    Orchestrates the split and transformation.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test (as numpy arrays or DFs), and the preprocessor.
    """
    # 1. Split Features and Target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Encode Target (Label Encoding for NN)
    # Map: Minor=0, PDO=1, Serious=2, Fatal=3
    # Note: We fit LabelEncoder to ensure consistent mapping
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # 3. Split Data (Train/Temp Split first)
    # First split: Train vs (Val + Test)
    # If val is 15% and test is 15%, then Temp is 30%.
    temp_size = val_size + test_size
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=temp_size, stratify=y_encoded, random_state=42
    )

    # Second split: Val vs Test (Equal split of the temp)
    # val_size / temp_size = 0.15 / 0.30 = 0.5
    relative_test_size = test_size / temp_size
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, stratify=y_temp, random_state=42
    )

    # 4. Build and Fit Preprocessor on TRAIN data only
    preprocessor = build_preprocessor(X_train_raw)
    preprocessor.fit(X_train_raw)

    # 5. Transform all sets
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Get feature names for interpretability later
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = None

    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "X_val": pd.DataFrame(X_val, columns=feature_names),
        "X_test": pd.DataFrame(X_test, columns=feature_names),
        "y_train": pd.DataFrame(y_train, columns=["target"]),
        "y_val": pd.DataFrame(y_val, columns=["target"]),
        "y_test": pd.DataFrame(y_test, columns=["target"]),
        "preprocessor": preprocessor,
        "label_encoder": le,
    }
