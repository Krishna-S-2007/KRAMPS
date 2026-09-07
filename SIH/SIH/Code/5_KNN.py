#########################################
# KNN
######################################### 

# Import libs
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
pipe_KNN_pca = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier())

# GridSearch & Cross-validation
# Setting parameters
param_grid = [{'kneighborsclassifier__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30],
               'kneighborsclassifier__algorithm': ['ball_tree', 'kd_tree', 'brute'],
               'kneighborsclassifier__leaf_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
               'kneighborsclassifier__metric': ['minkowski', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean']}]

# Set up grid search & cross validation for pca
gs_pca = GridSearchCV(estimator=pipe_KNN_pca, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)

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
