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
#file_path = r"E:\MindlessReading\Data\group_R_features_whole.csv"
#file_path = r"E:\MindlessReading\Data\group_R_features_last.csv"
file_path = '/Users/hsun11/Documents/GlassBrainLab/MindlessReading/Data/group_R_features_same-dur.csv'
df = pd.read_csv(file_path)

#%%
# change the labels to binary 1 and 0
# 1: MR
# 0: NR
#df['label'] = df['win_type'].replace({'MR': 1, 'NR': 0})
df['label'] = df['is_MWreported']
# throw out invalid samples
#df_valid = df.dropna(subset=['win_type'])


#%% EDA
# group by original
label_counts = df.groupby(['sub_id', 'reported_MW']).size().reset_index(name='Count')
label_counts['label'] = label_counts['reported_MW'].replace({0:'NR', 1:'self-reported'})
label_counts = label_counts.drop(columns=['reported_MW'])

# group by valid 
label_counts_ = df_valid.groupby(['sub_id', 'win_type']).size().reset_index(name='Count')
label_counts_ = label_counts_.rename(columns={'win_type':'label'})
label_counts_ = label_counts_[label_counts_['label'] != 'NR']

# vstack
lc = pd.concat([label_counts_, label_counts], axis=0)
lc['sub_id'] = lc['sub_id'].str[-3:]
# Create the bar plot
plt.figure(figsize=(15, 10))
sns.barplot(x='sub_id', y='Count', hue='label', data=lc, hue_order=['NR', 'self-reported','MR'])
# Customize the plot
plt.title('MR/NR Episodes for Each Subject')
plt.ylabel('Count')
plt.legend(title='Event Label')
plt.xlabel('Subject ID')
plt.grid()
# Show the plot
plt.show()


#%% EDA Last
# group by valid 
label_counts = df_valid.groupby(['sub_id', 'win_type']).size().reset_index(name='Count')
label_counts = label_counts.rename(columns={'win_type':'label'})

# vstack
lc = label_counts
lc['sub_id'] = lc['sub_id'].str[-3:]
# Create the bar plot
plt.figure(figsize=(15, 10))
sns.barplot(x='sub_id', y='Count', hue='label', data=lc, hue_order=['NR', 'MR'])
# Customize the plot
plt.title('MR/NR Episodes for Each Subject')
plt.ylabel('Count')
plt.legend(title='Event Label')
plt.xlabel('Subject ID')
plt.grid()
# Show the plot
plt.show()


#%% ML
# define features
features = ['pupil_slope', 'norm_pupil', 'norm_fix_word_num', 
            'norm_sac_num', 'norm_in_word_reg',
            'norm_out_word_reg', 'zscored_zipf_fixdur_corr',
            'norm_total_viewing', 'zscored_word_length_fixdur_corr']

# define models
models = {
        'Logistic Regression': LogisticRegression(),
        'Support Vector Machine': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier()
    }

# results dict
auroc_scores_dict = {
                    'Logistic Regression': [],
                    'Support Vector Machine': [],
                    'Decision Tree': [],
                    'Random Forest': [],
                    'AdaBoost': [],
                    'Linear Discriminant Analysis': [],
                    'KNN': [],
                    'Naive Bayes': [],
                    'XGBoost': []
                }

all_confusion_matrices_dict = {
                    'Logistic Regression': [],
                    'Support Vector Machine': [],
                    'Decision Tree': [],
                    'Random Forest': [],
                    'AdaBoost': [],
                    'Linear Discriminant Analysis': [],
                    'KNN': [],
                    'Naive Bayes': [],
                    'XGBoost': []
                }

true_labels_dict = {
                    'Logistic Regression': [],
                    'Support Vector Machine': [],
                    'Decision Tree': [],
                    'Random Forest': [],
                    'AdaBoost': [],
                    'Linear Discriminant Analysis': [],
                    'KNN': [],
                    'Naive Bayes': [],
                    'XGBoost': []
                }

predicted_probs_dict = {
                    'Logistic Regression': [],
                    'Support Vector Machine': [],
                    'Decision Tree': [],
                    'Random Forest': [],
                    'AdaBoost': [],
                    'Linear Discriminant Analysis': [],
                    'KNN': [],
                    'Naive Bayes': [],
                    'XGBoost': []
                }

df_cleaned = df.dropna(subset=features)
# run ML
for model_name, model in models.items():
    auroc_scores, all_confusion_matrices, true_labels, predicted_probs = supervised_ml_pipeline(df_cleaned, model, features=features, label='label', subject_column='sub_id')
    auroc_scores_dict[model_name] = auroc_scores
    all_confusion_matrices_dict[model_name] = all_confusion_matrices
    true_labels_dict[model_name] = true_labels
    predicted_probs_dict[model_name] = predicted_probs


#%% Plot ROC
plt.figure(figsize=(12,12))
bold_leg = []
index = 0
for model_name, _ in models.items():
    true_labels = true_labels_dict[model_name]
    predicted_probs = predicted_probs_dict[model_name]
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    auroc_score = roc_auc_score(true_labels, predicted_probs)
    if auroc_score > 0.9:
        bold_leg.append(index)
    plt.plot(fpr, tpr, label=f'{model_name} AUROC: {auroc_score:.2f}')
    index += 1

# annotate
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
leg = plt.legend(loc="lower right")
for index in bold_leg:
    leg.get_texts()[index].set_fontweight('bold')

plt.title('Average AUROC LOSOCV')


#%% Plot Feature Importance
# Random Forest
model = models['Random Forest']
# Get feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
# Plot feature importance
plt.figure(figsize=(12, 10))
plt.title("Random Forest Feature Importance")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Weight')
plt.show()

# XGBoost
model = models['XGBoost']
# Get feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
# Plot feature importance
plt.figure(figsize=(12, 10))
plt.title("XGBoost Feature Importance")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Weight')
plt.show()

# Navie Bayes
model = models['Naive Bayes']
# Get the log probabilities
log_prob = model.feature_log_prob_
# Calculate feature importance as the sum of log probabilities across classes
importance = np.sum(log_prob, axis=0)
indices = np.argsort(importances)[::-1]  # Sort feature importances in descending order
# Plot feature importance
plt.figure(figsize=(12, 10))
plt.title("Naive Bayes Feature Importance")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Weight')
plt.show()

# %% Plot Time Distribution
df_cleaned['win_label'] = df_cleaned['label'].replace({True: 'MR', False: 'NR'})
plt.figure()
sns.histplot(data=df_cleaned, x='win_dur', hue='win_label', kde=True)
