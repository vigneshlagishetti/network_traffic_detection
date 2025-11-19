"""Preprocessing utilities for NSL-KDD and CICIDS datasets.

Provides:
- load_nsl_kdd: loads NSL‑KDD train/test files and assigns column names
- build_preprocessing_pipeline: returns a sklearn Pipeline that encodes categorical
  features and scales numerical features
- resample_data: helpers for ADASYN and Borderline‑SMOTE resampling
- CLI to prepare processed CSVs into data/processed/

This module intentionally keeps imports local inside functions to allow the file
to be inspected or imported in environments where heavy deps are not yet installed.
"""
from typing import Tuple, Optional
import os


NSL_KDD_COLUMNS = [
	"duration",
	"protocol_type",
	"service",
	"flag",
	"src_bytes",
	"dst_bytes",
	"land",
	"wrong_fragment",
	"urgent",
	"hot",
	"num_failed_logins",
	"logged_in",
	"num_compromised",
	"root_shell",
	"su_attempted",
	"num_root",
	"num_file_creations",
	"num_shells",
	"num_access_files",
	"num_outbound_cmds",
	"is_host_login",
	"is_guest_login",
	"count",
	"srv_count",
	"serror_rate",
	"srv_serror_rate",
	"rerror_rate",
	"srv_rerror_rate",
	"same_srv_rate",
	"diff_srv_rate",
	"srv_diff_host_rate",
	"dst_host_count",
	"dst_host_srv_count",
	"dst_host_same_srv_rate",
	"dst_host_diff_srv_rate",
	"dst_host_same_src_port_rate",
	"dst_host_srv_diff_host_rate",
	"dst_host_serror_rate",
	"dst_host_srv_serror_rate",
	"dst_host_rerror_rate",
	"dst_host_srv_rerror_rate",
	"label",
]


def load_nsl_kdd(path: str):
	"""Load an NSL‑KDD file into a pandas DataFrame.

	The NSL‑KDD files typically have no header and are comma/space separated.
	We assign the canonical 41 feature names + label.
	"""
	try:
		import pandas as pd
	except Exception as e:  # pragma: no cover - dependency will be present in user env
		raise RuntimeError("pandas is required to load NSL‑KDD files") from e

	# read with no header; be permissive about separators
	# set low_memory=False to avoid mixed‑dtype warnings when reading large files
	df = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS, na_values=[""], skipinitialspace=True, low_memory=False)

	# Some mirrors include an extra column; drop any unnamed extras
	if df.shape[1] > len(NSL_KDD_COLUMNS):
		df = df.iloc[:, : len(NSL_KDD_COLUMNS)]

	return df


def build_preprocessing_pipeline(categorical_features=None, numeric_features=None):
	"""Construct a sklearn ColumnTransformer pipeline.

	- categorical_features: list of column names to one‑hot encode
	- numeric_features: list of numeric column names to scale

	Returns an sklearn Pipeline object.
	"""
	try:
		from sklearn.pipeline import Pipeline
		from sklearn.compose import ColumnTransformer
		from sklearn.preprocessing import OneHotEncoder, StandardScaler
		from sklearn.impute import SimpleImputer
	except Exception as e:  # pragma: no cover
		raise RuntimeError("scikit‑learn is required to build preprocessing pipeline") from e

	if categorical_features is None:
		categorical_features = ["protocol_type", "service", "flag"]
	if numeric_features is None:
		numeric_features = [c for c in NSL_KDD_COLUMNS if c not in categorical_features + ["label"]]

	# numeric pipeline: impute (median) -> scale
	numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

	# categorical pipeline: impute (constant) -> one‑hot (handle_unknown)
	# Use sparse_output=False for compatibility with newer scikit-learn versions
	categorical_pipeline = Pipeline(
		[("imputer", SimpleImputer(strategy="constant", fill_value="missing")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_features),
			("cat", categorical_pipeline, categorical_features),
		],
		remainder="drop",
	)

	full_pipeline = Pipeline([("preprocessor", preprocessor)])
	return full_pipeline


def resample_data(X, y, strategy: str = "adasyn", random_state: Optional[int] = 42):
	"""Resample data using ADASYN or Borderline‑SMOTE.

	X: numpy array or dataframe
	y: labels
	strategy: 'adasyn' or 'borderline'
	Returns resampled X_res, y_res
	"""
	try:
		if strategy == "adasyn":
			from imblearn.over_sampling import ADASYN

			sampler = ADASYN(random_state=random_state)
		elif strategy == "borderline":
			from imblearn.over_sampling import BorderlineSMOTE

			sampler = BorderlineSMOTE(random_state=random_state)
		else:
			raise ValueError("strategy must be 'adasyn' or 'borderline'")
	except Exception as e:  # pragma: no cover
		raise RuntimeError("imbalanced‑learn is required for resampling") from e

	X_res, y_res = sampler.fit_resample(X, y)
	return X_res, y_res


def prepare_and_save_nsl_kdd(train_path: str, test_path: str, out_dir: str = None):
	"""Load NSL‑KDD train/test, build and fit preprocessing pipeline on train, transform both
	and save processed arrays into data/processed.
	"""
	try:
		import pandas as pd
		import joblib
	except Exception as e:  # pragma: no cover
		raise RuntimeError("pandas and joblib are required to run prepare_and_save_nsl_kdd") from e

	if out_dir is None:
		out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
	os.makedirs(out_dir, exist_ok=True)

	print("Loading train data...")
	df_train = load_nsl_kdd(train_path)
	print("Loading test data...")
	df_test = load_nsl_kdd(test_path)

	# separate X/y
	y_train = df_train["label"].copy()
	X_train = df_train.drop(columns=["label"]).copy()

	y_test = df_test["label"].copy()
	X_test = df_test.drop(columns=["label"]).copy()

	# Define categorical features explicitly and coerce other columns to numeric where possible.
	categorical_features = ["protocol_type", "service", "flag"]
	# Coerce non-categorical columns to numeric to avoid imputer errors on strings
	for col in X_train.columns:
		if col not in categorical_features:
			X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
	for col in X_test.columns:
		if col not in categorical_features:
			X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

	print("Building preprocessing pipeline...")
	pipeline = build_preprocessing_pipeline()
	print("Fitting pipeline on train set (this may take a moment)...")
	pipeline.fit(X_train)

	print("Transforming datasets...")
	X_train_prep = pipeline.transform(X_train)
	X_test_prep = pipeline.transform(X_test)

	# Save pipeline for later
	pipeline_path = os.path.join(out_dir, "preprocessing_pipeline.joblib")
	joblib.dump(pipeline, pipeline_path)
	print(f"Saved pipeline to: {pipeline_path}")

	# Save processed arrays as npz for compactness
	train_out = os.path.join(out_dir, "train_processed.npz")
	test_out = os.path.join(out_dir, "test_processed.npz")

	import numpy as _np

	_np.savez_compressed(train_out, X=X_train_prep, y=_np.array(y_train))
	_np.savez_compressed(test_out, X=X_test_prep, y=_np.array(y_test))

	print(f"Saved processed train -> {train_out}")
	print(f"Saved processed test  -> {test_out}")


if __name__ == "__main__":
	# Simple CLI: find NSL‑KDD files in data/raw and prepare processed outputs
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
	raw_dir = os.path.join(root, "data", "raw")
	train_file = os.path.join(raw_dir, "KDDTrain+.txt")
	test_file = os.path.join(raw_dir, "KDDTest+.txt")
	if not os.path.exists(train_file) or not os.path.exists(test_file):
		print("Could not find NSL‑KDD train/test in data/raw. Run the download script first: python src/download_data.py")
	else:
		prepare_and_save_nsl_kdd(train_file, test_file)

