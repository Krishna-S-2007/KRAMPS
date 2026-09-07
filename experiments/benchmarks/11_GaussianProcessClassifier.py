#########################################
# GaussianProcessClassifier
######################################### 

# Import libs
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Import dataset
data = pd.read_csv('Dataset/Dataset_final.csv')

# Define X and y
X = data.drop('alert', axis=1)
y = data['alert']

# Apply SMOTE algorithm to balance dataset
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)

# # Standard Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_smote)

# # PCA
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# Perform split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Create pipeline
pipe_GPC_pca = make_pipeline(GaussianProcessClassifier(random_state=42))

# GridSearch & Cross-validation
# Setting parameters
param_grid = [{'gaussianprocessclassifier__max_iter_predict': [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]}]

# Set up grid search & cross validation for pca
gs_pca = GridSearchCV(estimator=pipe_GPC_pca, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

# Train grid search model
gs_pca = gs_pca.fit(X_train, y_train)

print('Best score from GridSearchCV: ' + str(gs_pca.best_score_))
print('Best parameters from GridSearchCV: ' + str(gs_pca.best_params_))

# # Logistic regression
# # Instantiate
# model = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', max_iter=1000, n_jobs=-1)

# # Train
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Print Matrices
# print('Misclassified samples: %d' % (y_test != y_pred).sum())
# print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
# print('Precision: %.4f' % precision_score(y_test, y_pred))
# print('Recall: %.4f' % recall_score(y_test, y_pred))
# print('F1: %.4f' % f1_score(y_test, y_pred))
# print('ROC AUC: %.4f' % roc_auc_score(y_test, y_pred))

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot non-normalized confusion matrix
# plt.figure(figsize = (10,10))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['benign', 'suspicious'])
# disp.plot()
# plt.show()
