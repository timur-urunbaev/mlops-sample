import mlflow
import polars as pl

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

mlflow.set_tracking_uri(uri='http://localhost:5000')
mlflow.set_experiment("Alif.HW")

ratings = pl.read_csv('ml-100k/u.data', separator='\t', has_header=False, new_columns=['user_id', 'item_id', 'rating', 'timestamp'])
users = pl.read_csv('ml-100k/u.user', separator='|', has_header=False, new_columns=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
movies = pl.read_csv('ml-100k/u.item', separator='|', has_header=False, encoding='latin-1', new_columns=[
    'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
    'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
])

data = ratings.join(users, on='user_id').join(movies, on='item_id')

data = data.with_columns((data['rating'] == 5).cast(pl.Int32).alias('is_rating_5'))

data = data.drop(['video_release_date'])
data = data.drop_nulls()

feature_columns = ['age', 'gender', 'occupation'] + movies.columns[5:]
X = data.select(feature_columns)
y = data['is_rating_5']

# Encode categorical variables
le_gender = LabelEncoder()
X = X.with_columns(pl.Series('gender', le_gender.fit_transform(X['gender'].to_list())))

le_occupation = LabelEncoder()
X = X.with_columns(pl.Series('occupation', le_occupation.fit_transform(X['occupation'].to_list())))

# Convert Polars DataFrames to NumPy arrays for sklearn
X_np = X.to_numpy()
y_np = y.to_numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

params = {
    "n_estimators": 100,
    "random_state": 42
}

with mlflow.start_run():
    mlflow.set_tag("Training Info", "Baseline", "HW Alif")
    mlflow.log_params(params=params)

    clf = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        random_state=params['random_state']
    )
    clf.fit(X_resampled, y_resampled)

    mlflow.sklearn.log_model(clf, "random_forest_model")
    signature = mlflow.models.infer_signature(X_train, clf.predict(X_train))

    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['1']['precision'])
    mlflow.log_metric("recall", report['1']['recall'])
    mlflow.log_metric("f1_score", report['1']['f1-score'])

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
