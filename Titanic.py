import os
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotext as plt_ext
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def extract_title(name):
    title_search = re.search(r' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def preprocess_data(df):
    df = df.copy()
    
    # 1. Extract Title
    df['Title'] = df['Name'].apply(extract_title)
    
    # Replace rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # 2. Family Size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # 3. Cabin Presence
    df['Has_Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    
    # Fill missing Age by Title median
    title_age_median = df.groupby('Title')['Age'].median()
    df['Age'] = df.apply(lambda row: title_age_median[row['Title']] if pd.isna(row['Age']) else row['Age'], axis=1)
    
    # Impute missing Fare with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Impute missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=drop_cols)
    
    return df

def build_pipeline():
    categorical_features = ['Sex', 'Embarked', 'Title']
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Has_Cabin']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    
    return clf, numeric_features, categorical_features


def save_plot(filename):
    output_dir = 'images'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Saved image to '{filepath}'")
    return filepath


def main():
    # Load dataset
    df = pd.read_csv('titanic.csv')
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Define features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build and train pipeline
    pipeline, num_features, cat_features = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("Model Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_plot('confusion_matrix.png')
    
    # Feature Importance
    clf = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names = num_features + list(cat_feature_names)
    
    importances = clf.feature_importances_
    
    # Save feature importance plot (horizontal)
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, color='#2c7bb6')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    save_plot('feature_importance.png')
    print()
    
    # Display feature importance in the terminal
    plt_ext.clear_figure()
    plt_ext.title("Feature Importances")
    names = [feature_names[i] for i in indices]
    vals = importances[indices].tolist()
    plt_ext.bar(names, vals)
    plt_ext.show()
    print("\n")
    
    # Try SHAP
    try:
        import shap
        print("Generating SHAP values...")
        # Since SHAP expects arrays or dataframes, we can transform X_train
        # Note: If memory becomes an issue, we can sample the background dataset
        X_train_transformed = preprocessor.transform(X_train)
        
        # Determine if it's a sparse matrix and convert to dense if necessary for TreeExplainer
        if hasattr(X_train_transformed, "toarray"):
            X_train_transformed = X_train_transformed.toarray()
            
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_train_transformed)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        plt.figure()
        shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, show=False)
        save_plot('shap_summary.png')
    except ImportError:
        print("SHAP not installed. Skipping SHAP summary plot.")
    except Exception as e:
        print(f"Error generating SHAP values: {e}")
    
    # Save model
    joblib.dump(pipeline, 'titanic_model.pkl')
    print("Model saved to 'titanic_model.pkl'")

if __name__ == "__main__":
    main()
