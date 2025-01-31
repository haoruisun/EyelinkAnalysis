#%% Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve


#%% Functions
def supervised_ml_pipeline(data, model, features, label, subject_column):
    # Prepare the cross-validation strategy (Leave-One-Subject-Out)
    logo = LeaveOneGroupOut()

    # Extract features, labels, and subject IDs
    X = data[features].values
    y = data[label].values
    groups = data[subject_column].values

    # Lists to store results
    auroc_scores = []
    all_confusion_matrices = []
    true_labels = []
    predicted_probs = []
    
    # Leave-one-subject-out cross-validation
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the model
        model.fit(X_train, y_train)

        # Get predictions
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate AUROC for the current fold
        auroc = roc_auc_score(y_test, y_pred_prob)
        auroc_scores.append(auroc)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        all_confusion_matrices.append(conf_matrix)

        # Store true labels and predicted probabilities
        true_labels.extend(y_test)
        predicted_probs.extend(y_pred_prob)


    return auroc_scores, all_confusion_matrices, true_labels, predicted_probs

def plot_roc(true_labels, predicted_probs):
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_score(true_labels, predicted_probs):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")



  # Calculate average AUROC
    # avg_auroc = np.mean(auroc_scores)
    # print(f"Average AUROC: {avg_auroc}")

    # # Compute overall confusion matrix
    # total_conf_matrix = sum(all_confusion_matrices)
    # print("Overall Confusion Matrix:")
    # print(total_conf_matrix)

#%% Load Dataset
file_path = '../../res/s10014_L_features_slide.csv'
df = pd.read_csv(file_path)

#%% Get training data
df_train = df[(df['label']!='self_report') & (df['relative_time']==0)]
