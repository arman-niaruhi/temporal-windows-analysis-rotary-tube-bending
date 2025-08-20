from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1️⃣ Load the Wine dataset
data = load_wine()
X = data.data
y = data.target

# 2️⃣ Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3️⃣ Create a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),        # Standardize features
    ('pca', PCA(n_components=2)),       # Reduce to 2 principal components
    ('classifier', LogisticRegression()) # Logistic Regression
])

# 4️⃣ Fit the pipeline on training data
pipeline.fit(X_train, y_train)

# 5️⃣ Predict on test data
y_pred = pipeline.predict(X_test)

# 6️⃣ Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
