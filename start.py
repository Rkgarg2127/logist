from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def displayData(data):
    # Display the first few rows
    print("Dataset Preview:")
    print(data.head())
    # Display summary of the dataset
    print("\nDataset Info:")
    print(data.info())
    # Check for missing values
    print("\nMissing Values Count:")
    print(data.isnull().sum())
    # Check basic statistics of numeric columns
    print("\nStatistical Summary:")
    print(data.describe())

def cleanData(data):
    # Drop rows with missing target values
    data = data.dropna(subset=["Delayed"])
    # Fill missing values in other columns (if any)
    data.fillna({
        "Weather Conditions": "Clear",
        "Traffic Conditions": "Moderate",
        "Distance (km)": data["Distance (km)"].mean()
    }, inplace=True)
    print("\nAfter Cleaning, Missing Values Count:")
    print(data.isnull().sum())
    # Convert categorical variables to dummy/encoded variables
    data = pd.get_dummies(data, columns=[
        "Vehicle Type", "Weather Conditions", "Traffic Conditions", "Origin", "Destination"
    ], drop_first=True)
    return data

def GridSearchCVModel(X_train, X_test, y_train, y_test):
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Save the scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")

    # Define the parameter grid for logistic regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'solver': ['lbfgs', 'liblinear']  # Different solvers
    }
    # Initialize GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=5, scoring='accuracy')
    # Fit the grid search to the training data
    grid_search.fit(X_train_scaled, y_train)
    # Get the best parameters from the grid search
    best_params = grid_search.best_params_
    print(f"\nBest Parameters: {best_params}")
    # Train the model with the best parameters
    model_tuned = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=2000)
    model_tuned.fit(X_train_scaled, y_train)
    return model_tuned

def RandomForestClassifierModel(X_train, X_test, y_train, y_test):
    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model    

def GradientBoostingClassifierModel(X_train, X_test, y_train, y_test):
    # Initialize and train the Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    # Evaluate the Gradient Boosting model
    return gb_model;

def printModelMetrics(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Feature Importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        print("\nFeature Importance:")
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print(feature_importances.sort_values(ascending=False))

def saveModel(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")


# Load the Excel file
data = pd.read_excel("/Users/rkgarg2127/Desktop/python/logist/shipment_data.xlsx")

# Display the dataset
displayData(data)

# Clean the dataset
data = cleanData(data)

print("\nDataset After Encoding:")
print(data.head())

# Target variable: 'Delayed' column encoded as binary (1 for Yes, 0 for No)
y = data["Delayed"].apply(lambda x: 1 if x == "Yes" else 0)

# Features (drop non-feature columns)
X = data.drop(columns=["Delayed", "Shipment ID", "Shipment Date", "Planned Delivery Date", "Actual Delivery Date"])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")

# Train and evaluate a Logistic Regression model
print("\nLogistic Regression Model:")
gridSearchCVModel= GridSearchCVModel(X_train, X_test, y_train, y_test)
printModelMetrics(gridSearchCVModel, X_train, X_test, y_train, y_test)
saveModel(gridSearchCVModel, "logistic_regression_model.pkl")


# Train and evaluate a Random Forest Classifier model
print("\nRandom Forest Classifier Model:")
randomForestClassifierModel= RandomForestClassifierModel(X_train, X_test, y_train, y_test)
printModelMetrics(randomForestClassifierModel, X_train, X_test, y_train, y_test)
saveModel(randomForestClassifierModel, "random_forest_model.pkl")

# Train and evaluate a Gradient Boosting Classifier model
print("\nGradient Boosting Classifier Model:")
gradientBoostingClassifier= GradientBoostingClassifierModel(X_train, X_test, y_train, y_test)
printModelMetrics(gradientBoostingClassifier, X_train, X_test, y_train, y_test)
saveModel(gradientBoostingClassifier, "gradient_boosting_model.pkl")

