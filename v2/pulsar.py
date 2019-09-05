import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
# RandomizedSearchCV for tuning RF params
from sklearn.model_selection import RandomizedSearchCV


# Evaluations
from sklearn.metrics import classification_report, confusion_matrix
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# SGDClassifier
from sklearn.linear_model import SGDClassifier
# LinearSVC
from sklearn.svm import LinearSVC
# Cross Val
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, cross_val_predict
# PCA
from sklearn.decomposition import PCA
# Outliers search
from sklearn.neighbors import LocalOutlierFactor
# GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

stars = pd.read_csv(r'../predicting-a-pulsar-star/pulsar_stars.csv')

# Cename the columns
stars.columns = ['MOTIP',
'SDOTIP',
'EKOTIP',
'SOTIP',
'MOTDSC',
'SEOTDSC',
'EKOTDSC',
'SOTDSC',
'target_class']

print('Update plots? [y/n]')
# input
todo_plots = input()

if (todo_plots == 'y'):
    # Unscaled Data Summary
    plt.figure(figsize=(12,8))
    sns.heatmap(stars.describe()[1:].transpose(),
                annot=True,linecolor="w",
                linewidth=2,cmap=sns.color_palette("Set2"))
    plt.title("Data summary")
    # plt.show()
    plt.savefig("summary.png")
    plt.clf()

# Scaling the dataset
scaled_features = stars.copy()
col_names = stars.columns[:-1]
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
# print(scaled_features)

stars = scaled_features

# if (todo == 'y'):

print(stars.describe())

# Checking the presence of null values in the data set
print(stars.isnull().any())

if (todo_plots == 'y'):
    # Data Summary
    plt.figure(figsize=(12,8))
    sns.heatmap(stars.describe()[1:].transpose(),
                annot=True,linecolor="w",
                linewidth=2,cmap=sns.color_palette("Set2"))
    plt.title("Data summary")
    # plt.show()
    plt.savefig("heatmap.png")
    plt.clf()

    # Check correlation between variables
    correlation = stars.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation,annot=True,
                cmap=sns.color_palette("magma"),
                linewidth=2,edgecolor="k")
    plt.title("Variables correlation")
    # plt.show()
    plt.savefig("correlation.png")
    plt.clf()

    # Check targets number
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    ax = sns.countplot(y = stars["target_class"],
                    palette=["r","g"],
                    linewidth=1,
                    edgecolor="k"*2)
    for i,j in enumerate(stars["target_class"].value_counts().values):
        ax.text(.7,i,j,weight = "bold",fontsize = 27)
    plt.title("Count for pulsars in the datset")

    plt.subplot(122)
    plt.pie(stars["target_class"].value_counts().values,
            labels=["not pulsar stars","pulsar stars"],
            autopct="%1.0f%%",wedgeprops={"linewidth":2,"edgecolor":"white"})
    my_circ = plt.Circle((0,0),.7,color = "white")
    plt.gca().add_artist(my_circ)
    plt.subplots_adjust(wspace = .2)
    plt.title("Proportion of pulsars in the dataset")
    # plt.show()
    plt.savefig("proportion.png")
    plt.clf()

    # Pairs plot
    # sns.pairplot(stars,hue="target_class")
    # plt.title("pair plot for variables")
    # plt.show()

    # palette2 = sns.color_palette(["#bbbbbb", "#a800a2"])

    # pg = sns.PairGrid(stars, hue = "target_class", hue_order = [0, 1], vars = stars.columns)
    pg = sns.pairplot(stars, hue = "target_class", hue_order = [0, 1], vars = stars.columns, height=10)
    pg.map_diag(sns.kdeplot),
    pg.map_offdiag(plt.scatter, s = 1, alpha = 0.2)
    pg.savefig("pairs.png")
    plt.clf()

# PCA
groups = stars.groupby(['target_class'])
nonPulsarsG = groups.get_group(0)
pulsarsG = groups.get_group(1)

pca = PCA(n_components = 2, random_state = 0)
pca.fit(stars.filter(regex = "[^target_class]").values)
nonPulsarComponents = pca.transform(nonPulsarsG.filter(regex = "[^target_class]").values)
pulsarComponents = pca.transform(pulsarsG.filter(regex = "[^target_class]").values)

if (todo_plots == 'y'):
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(111)
    ax1.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], s = 10, label = 'Non-Pulsars')
    ax1.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], s = 10, label = 'Pulsars')
    ax1.axis('tight')
    plt.legend(loc = 'lower left')
    plt.savefig("pca.png")
    plt.clf()

# Search for Outlier

# Clearing Non pulsars

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(nonPulsarComponents)

X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
# radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
radius[radius < 0.2] = 0

if (todo_plots == 'y'):
    plt.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], s=1000 * radius, edgecolors='r',
                facecolors='none', label='Outlier scores')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]

    plt.savefig("OutlierNonP.png")
    plt.clf()

# Clearing the dataset

outlier_positions = np.where(radius > 0)[0]
outlier_indexes = list(map(lambda position: nonPulsarsG.index[position], outlier_positions))
nonPulsarsClear = nonPulsarsG.drop(outlier_indexes)

# Clearing pulsars

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
y_pred = clf.fit_predict(pulsarComponents)

X_scores = clf.negative_outlier_factor_

plt.title("Local Outlier Factor (LOF)")
plt.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], color='k', s=3., label='Data points')
# plot circles with radius proportional to the outlier scores
# radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
radius[radius < 0.2] = 0

if (todo_plots == 'y'):
    plt.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], s=1000 * radius, edgecolors='r',
                facecolors='none', label='Outlier scores')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))

    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]

    plt.savefig("OutlierP.png")
    plt.clf()

# Clearing the dataset

outlier_positions = np.where(radius > 0)[0]
outlier_indexes = list(map(lambda position: pulsarsG.index[position], outlier_positions))
PulsarsClear = pulsarsG.drop(outlier_indexes)

# Full cleared dataset

cleared_full_data = pd.concat([nonPulsarsClear, PulsarsClear])

# Dropping week features

cleared_dropped_full_data = cleared_full_data.drop('SOTIP', axis=1)
cleared_dropped_full_data = cleared_dropped_full_data.drop('SOTDSC', axis=1)

cleared_full_data = cleared_dropped_full_data

# Splitting dataset

# Remove target column
def data_prep(df):
    feature_columns = df.columns[:-1]
    df_features = pd.DataFrame(data=df, columns=feature_columns)
    return df_features

# Calling the function data_prep to get feature's columns
df_features = data_prep(cleared_full_data)

# Spiting the data to train and test the model
X = df_features.copy()
y = cleared_full_data['target_class'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# MODELS===============================================

print('Test all models? [y/n]')
# input
todo_models = input()

if (todo_models == 'y'):

    # RF===============================================

    print('RandomForestClassifier: \n')

    rf_model = RandomForestClassifier(n_estimators=200, random_state = 0)
    rf_model.fit(X_train, y_train)

    # Prediction using Random Forest Model
    rf_prediction = rf_model.predict(X_test)

    # Evaluations
    print('Classification Report: \n')
    print(classification_report(y_test, rf_prediction))
    print('\nConfusion Matrix: \n')
    print(confusion_matrix(y_test, rf_prediction))

    # Cross validation
    print('\nCross validation: \n')
    recall_estimate = cross_val_score(rf_model, X_test, y_test, cv=10, scoring='recall_weighted')
    print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

    y_pred = cross_val_predict(rf_model, X_test, y_test, cv=10)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

    # LinearSVC===============================================

    print('LinearSVC: \n')

    linear_svc = LinearSVC(random_state = 0)
    linear_svc.fit(X_train, y_train)

    # Prediction using LinearSVC
    linear_svc_prediction = linear_svc.predict(X_test)

    # Evaluations
    print('Classification Report: \n')
    print(classification_report(y_test, linear_svc_prediction))
    print('\nConfusion Matrix: \n')
    print(confusion_matrix(y_test, linear_svc_prediction))

    # Cross validation
    print('\nCross validation: \n')
    recall_estimate = cross_val_score(linear_svc, X_test, y_test, cv=10, scoring='recall_weighted')
    print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

    y_pred = cross_val_predict(linear_svc, X_test, y_test, cv=10)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

    # SGDClassifier===============================================

    print('SGDClassifier: \n')

    sgd = SGDClassifier(random_state = 0)
    sgd.fit(X_train, y_train)

    # Prediction using SGDClassifier
    sgd_prediction = sgd.predict(X_test)

    # Evaluations
    print('Classification Report: \n')
    print(classification_report(y_test, sgd_prediction))
    print('\nConfusion Matrix: \n')
    print(confusion_matrix(y_test,sgd_prediction))

    # Cross validation
    print('\nCross validation: \n')
    recall_estimate = cross_val_score(sgd, X_test, y_test, cv=10, scoring='recall_weighted')
    print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

    y_pred = cross_val_predict(sgd, X_test, y_test, cv=10)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

    # GradientBoostingClassifier===============================================

    print('GradientBoostingClassifier: \n')

    gradient_boosting = GradientBoostingClassifier(random_state = 0)
    gradient_boosting.fit(X_train, y_train)

    # Prediction using GradientBoostingClassifier
    gradient_boosting_prediction = gradient_boosting.predict(X_test)

    # Evaluations
    print('Classification Report: \n')
    print(classification_report(y_test, gradient_boosting_prediction))
    print('\nConfusion Matrix: \n')
    print(confusion_matrix(y_test, gradient_boosting_prediction))

    # Cross validation
    print('\nCross validation: \n')
    recall_estimate = cross_val_score(gradient_boosting, X_test, y_test, cv=10, scoring='recall_weighted')
    print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

    y_pred = cross_val_predict(gradient_boosting, X_test, y_test, cv=10)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(conf_mat)

# Selecting RF, let's tune it===============================================

# print('\nTuning RF:')

# #fit random forest
# forest = RandomForestClassifier(random_state = 0)
# forest.fit(X_train, y_train)

# param_grid = [
#     {'n_estimators': [115, 120, 125], 'max_features': [4, 5], 
#     'max_depth': range(1,20), 'bootstrap': [True, False], 'class_weight': ['balanced', 'balanced_subsample']}
# ]

# # grid_search_forest = GridSearchCV(forest, param_grid, cv=10, scoring='neg_mean_squared_error')
# grid_search_forest = GridSearchCV(forest, param_grid, cv=10, scoring='recall_weighted')
# grid_search_forest.fit(X_train, y_train)

# #now let's how the RMSE changes for each parameter configuration
# # cvres = grid_search_forest.cv_results_
# # for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
# #     print(np.sqrt(-mean_score), params)

# grid_best = grid_search_forest.best_estimator_

# #find the best model of grid search
# print('\nbest model:')
# print(grid_best)

# # From GridSearchCV we have:

# RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
#                        criterion='gini', max_depth=15, max_features=4,
#                        max_leaf_nodes=None, min_impurity_decrease=0.0,
#                        min_impurity_split=None, min_samples_leaf=1,
#                        min_samples_split=2, min_weight_fraction_leaf=0.0,
#                        n_estimators=115, n_jobs=None, oob_score=False,
#                        random_state=0, verbose=0, warm_start=False)

# forest = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
#                        criterion='gini', max_depth=15, max_features=4,
#                        max_leaf_nodes=None, min_impurity_decrease=0.0,
#                        min_impurity_split=None, min_samples_leaf=1,
#                        min_samples_split=40, min_weight_fraction_leaf=0.0,
#                        n_estimators=120, n_jobs=None, oob_score=False,
#                        random_state=0, verbose=0, warm_start=False)

print('\nTesting best model:')

forest = RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
                       criterion='gini', max_depth=15, max_features=4,
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=40, min_weight_fraction_leaf=0.0,
                       n_estimators=120, n_jobs=None, oob_score=False,
                       random_state=0, verbose=0, warm_start=False)

forest.fit(X_train, y_train)

# Prediction using Random Forest Model
rf_prediction = forest.predict(X_test)

# Evaluations
print('Classification Report: \n')
print(classification_report(y_test, rf_prediction))
print('\nConfusion Matrix: \n')
print(confusion_matrix(y_test, rf_prediction))

recall_estimate = cross_val_score(forest, X_test, y_test, cv=10, scoring='recall_weighted')
print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

y_pred = cross_val_predict(forest, X_test, y_test, cv=10)
conf_mat = confusion_matrix(y_test, y_pred)



# Show confusion matrix in a separate window
# thresh = conf_mat.max() / 1.5
plt.matshow(conf_mat)
plt.title('Confusion matrix')
plt.colorbar(cmap="BuPu", alpha=0.4)
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(2):
    for j in range(2):
        # plt.text(j,i, str(conf_mat[i][j])[:6])
        plt.text(j, i, str(conf_mat[i][j])[:6],
                     horizontalalignment="center",
                     color="red")
plt.savefig("matrix.png")
plt.clf()

# thresh = conf_mat.max() / 2
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat = conf_mat/conf_mat.astype(np.float).sum(axis=1)
# Show confusion matrix in a separate window
plt.matshow(conf_mat)
plt.title('Confusion matrix')
plt.colorbar(cmap="BuPu", alpha=0.4)
plt.ylabel('True label')
plt.xlabel('Predicted label')
for i in range(2):
    for j in range(2):
        # plt.text(j,i, str(conf_mat[i][j])[:6])
        plt.text(j, i, str(conf_mat[i][j])[:6],
                     horizontalalignment="center",
                     color="red")
                     
plt.savefig("matrixN.png")
plt.clf()

print(conf_mat)

print('Save this model? [y/n]')
# input
todo_save = input()

if (todo_save == 'y'):
    with open('pulsarmodel', 'wb') as f:
        pickle.dump(forest, f)

    print("Model saved")



