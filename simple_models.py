import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, f1_score, classification_report, confusion_matrix, accuracy_score

from preprocess import DataPreprocessor


preprocessor = DataPreprocessor('../data/video_games.csv')
preprocessor.preprocess()
data = preprocessor.df
# print(data['Metadata'])
# print(data.columns)
# print(data['Features.Online?'].value_counts())
features_to_test = [
    # 'Features.Max Players', 'Metadata.Publishers', 'Release.Rating','Release.Year',
                    'Metrics.Used Price', 'Release.Console', 'Length.Main + Extras.Average', 'Length.Main + Extras.Leisure', 'Length.Main + Extras.Median',
                    'Length.Main Story.Average', 'Length.Completionists.Average', 'Metrics.Review Score', 'Metrics.Sales']

features_selected = ['Metadata.Publishers', 'Metrics.Used Price', 
                     'Length.Main Story.Average', 'Length.Completionists.Average', 'Length.Main + Extras.Leisure']
target1 = ['Metrics.Review Score']
target2 = ['Metrics.Sales']
# print(data[:10]['Metadata.Genres'])
print(data['Metrics.Review Score'].describe())
print(data['Metrics.Sales'].describe())

# correlation_matrix = data[features_to_test].corr()

# # Test multicollinearity and drop highly correlated features
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix')
# plt.show()

X = data[features_selected]
y = data[target1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ************  Random Forest model  ************

# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(X_train, y_train)

# # Calculate permutation importances
# perm_importance = permutation_importance(rf_regressor, X_test, y_test, n_repeats=10, random_state=42)

# # Plotting permutation importances
# sorted_idx = perm_importance.importances_mean.argsort()
# plt.figure(figsize=(10, 6))
# plt.barh(range(X.shape[1]), perm_importance.importances_mean[sorted_idx], align='center')
# plt.yticks(range(X.shape[1]), [features_selected[i] for i in sorted_idx])
# plt.xlabel('Permutation Importance')
# plt.title('Permutation Importance for Features')
# plt.show()

# ************  Decision tree regression model  ************
# tree_regressor = DecisionTreeRegressor(random_state=42)
# tree_regressor.fit(X_train, y_train)

# y_pred = tree_regressor.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared Score: {r2}")

# plt.figure(figsize=(10, 6))
# plot_tree(tree_regressor, filled=True, feature_names=features_selected)
# plt.show()

# ************  Linear Regression model  ************

# linear_regressor = LinearRegression()
# linear_regressor.fit(X_train, y_train)

# y_pred = linear_regressor.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error: {mse}")
# print(f"R-squared Score: {r2}")


# ***************************  TURN INTO BINARY CLASSIFICATION  ****************************
sale_threshold = -0.1
score_threshold = 70 
data['good_sale'] = data['Metrics.Sales'].apply(lambda x: 'Yes' if x >= sale_threshold else 'No')
data['good_score'] = data['Metrics.Review Score'].apply(lambda x: 'Yes' if x >= score_threshold else 'No')
# print(data.head())


target_sale = data['good_sale']  # Target variable for good_sale prediction
target_score = data['good_score']  # Target variable for good_score prediction

# Splitting for good_sale prediction
X_train_sale, X_test_sale, y_train_sale, y_test_sale = train_test_split(data[features_selected], target_sale, test_size=0.2, random_state=42)

# Splitting for good_score prediction
X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(data[features_selected], target_score, test_size=0.2, random_state=42)

# *************** Logistic Regression ***************
# logreg_sale = LogisticRegression(random_state=42)
# logreg_sale.fit(X_train_sale, y_train_sale)

# logreg_score = LogisticRegression(random_state=42)
# logreg_score.fit(X_train_score, y_train_score)

# y_pred_sale = logreg_sale.predict(X_test_sale)
# y_pred_score = logreg_score.predict(X_test_score)

# accuracy_sale = accuracy_score(y_test_sale, y_pred_sale)
# accuracy_score = accuracy_score(y_test_score, y_pred_score)

# print("Accuracy for good_sale prediction:", accuracy_sale)
# print("Accuracy for good_score prediction:", accuracy_score)

# # Display classification reports and confusion matrices
# print("\nClassification Report for good_sale prediction:")
# print(classification_report(y_test_sale, y_pred_sale))

# print("\nClassification Report for good_score prediction:")
# print(classification_report(y_test_score, y_pred_score))

# print("\nConfusion Matrix for good_sale prediction:")
# print(confusion_matrix(y_test_sale, y_pred_sale))

# print("\nConfusion Matrix for good_score prediction:")
# print(confusion_matrix(y_test_score, y_pred_score))

# *************** Decision Tree and Random Forest ***************

dt_classifier_sale = DecisionTreeClassifier(random_state=42)
dt_classifier_sale.fit(X_train, y_train_sale)

rf_classifier_sale = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_sale.fit(X_train, y_train_sale)

dt_classifier_score = DecisionTreeClassifier(random_state=42)
dt_classifier_score.fit(X_train, y_train_score)

rf_classifier_score = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_score.fit(X_train, y_train_score)

y_pred_dt_sale = dt_classifier_sale.predict(X_test)
y_pred_rf_sale = rf_classifier_sale.predict(X_test)
y_pred_dt_score = dt_classifier_score.predict(X_test)
y_pred_rf_score = rf_classifier_score.predict(X_test)

accuracy_dt_sale = accuracy_score(y_test_sale, y_pred_dt_sale)
accuracy_rf_sale = accuracy_score(y_test_sale, y_pred_rf_sale)
accuracy_dt_score = accuracy_score(y_test_score, y_pred_dt_score)
accuracy_rf_score = accuracy_score(y_test_score, y_pred_rf_score)

print(f"Accuracy of Decision Tree Classifier for good_sale: {accuracy_dt_sale}")
print(f"Accuracy of Random Forest Classifier for good_sale: {accuracy_rf_sale}")
print(f"Accuracy of Decision Tree Classifier for good_score: {accuracy_dt_score}")
print(f"Accuracy of Random Forest Classifier for good_score: {accuracy_rf_score}")

dt_feature_importance_sale = dt_classifier_sale.feature_importances_
dt_feature_importance_scores_sale = dict(zip(features_selected, dt_feature_importance_sale))
print("Feature Importance for good_sale (Decision Tree):")
print(dt_feature_importance_scores_sale)

# Get feature importance from Decision Tree Classifier for good_score
dt_feature_importance_score = dt_classifier_score.feature_importances_
dt_feature_importance_scores_score = dict(zip(features_selected, dt_feature_importance_score))
print("Feature Importance for good_score (Decision Tree):")
print(dt_feature_importance_scores_score)

rf_feature_importance_sale = rf_classifier_sale.feature_importances_
rf_feature_importance_scores_sale = dict(zip(features_selected, rf_feature_importance_sale))
print("Feature Importance for good_sale (Random Forest):")
print(rf_feature_importance_scores_sale)

# Get feature importance from Random Forest Classifier for good_score
rf_feature_importance_score = rf_classifier_score.feature_importances_
rf_feature_importance_scores_score = dict(zip(features_selected, rf_feature_importance_score))
print("Feature Importance for good_score (Random Forest):")
print(rf_feature_importance_scores_score)