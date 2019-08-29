import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Evaluations
from sklearn.metrics import classification_report,confusion_matrix
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# SGDClassifier
from sklearn.linear_model import SGDClassifier
# LinearSVC
from sklearn.svm import LinearSVC
# Cross Val
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, cross_val_predict

# Remove target column
def data_prep(df):
    feature_columns = df.columns[:-1]
    df_features = pd.DataFrame(data=df, columns=feature_columns)
    return df_features

# Scaler function
def standardScaling(feature):
   scaler = StandardScaler().fit(feature)
   scaled_feature = scaler.transform(feature)
   scaled_feat = pd.DataFrame(data = scaled_feature, columns = df_features.columns)
   return scaled_feat

stars = pd.read_csv(r'./predicting-a-pulsar-star/pulsar_stars.csv')

# Cename the columns
stars.columns = ['Mean_of_the_integrated_profile',
'Standard_deviation_of_the_integrated_profile',
'Excess_kurtosis_of_the_integrated_profile',
'Skewness_of_the_integrated_profile',
'Mean_of_the_DM_SNR_curve',
'Standard_eviation_of_the_DM_SNR_curve',
'Excess_kurtosis_of_the_DM_SNR_curve',
'Skewness_of_the_DM_SNR_curve',
'target_class']


print('Show plots? [y/n]')
# input
todo = input()

if (todo == 'y'):

    # Checking the presence of null values in the data set
    print(stars.isnull().any())

    # Data Summary
    plt.figure(figsize=(12,8))
    sns.heatmap(stars.describe()[1:].transpose(),
                annot=True,linecolor="w",
                linewidth=2,cmap=sns.color_palette("Set2"))
    plt.title("Data summary")
    plt.show()

    # Check correlation between variables
    correlation = stars.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(correlation,annot=True,
                cmap=sns.color_palette("magma"),
                linewidth=2,edgecolor="k")
    plt.title("Variables correlation")
    plt.show()

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
    plt.show()

    # Pairs plot
    sns.pairplot(stars,hue="target_class")
    plt.title("pair plot for variables")
    plt.show()
                              
# Calling the function data_prep to get feature's columns
df_features = data_prep(stars)

# Spiting the data to train and test the model
X = df_features.copy()
y = stars['target_class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Calling the scaler function by passing X_train and X_test to get the scaled data set
X_train_scaled = standardScaling(X_train)
X_test_scaled = standardScaling(X_test)
X_full_scaled = standardScaling(X)

# During scaling process the index of X_train and X_test are changed.
# So, you must also reset the index of y_train and y_test as below, otherwise you will get index mismatch error.
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# MODELS===============================================

# RF===============================================

print('RandomForestClassifier: \n')

rf_model = RandomForestClassifier(n_estimators=200)
rf_model.fit(X_train_scaled, y_train)

# Prediction using Random Forest Model
rf_prediction = rf_model.predict(X_test_scaled)

# Evaluations
print('Classification Report: \n')
print(classification_report(y_test, rf_prediction))
print('\nConfusion Matrix: \n')
print(confusion_matrix(y_test, rf_prediction))

# Cross validation
print('\nCross validation: \n')
recall_estimate = cross_val_score(rf_model, X_test_scaled, y_test, cv=10, scoring='recall_weighted')
print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

y_pred = cross_val_predict(rf_model, X_test_scaled, y_test, cv=10)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)

# LinearSVC===============================================

print('LinearSVC: \n')

linear_svc = LinearSVC()
linear_svc.fit(X_train_scaled, y_train)

# Prediction using LinearSVC
linear_svc_prediction = linear_svc.predict(X_test_scaled)

# Evaluations
print('Classification Report: \n')
print(classification_report(y_test, linear_svc_prediction))
print('\nConfusion Matrix: \n')
print(confusion_matrix(y_test, linear_svc_prediction))

# Cross validation
print('\nCross validation: \n')
recall_estimate = cross_val_score(linear_svc, X_test_scaled, y_test, cv=10, scoring='recall_weighted')
print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

y_pred = cross_val_predict(rf_model, X_test_scaled, y_test, cv=10)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)

# SGDClassifier===============================================

print('SGDClassifier: \n')

sgd = SGDClassifier()
sgd.fit(X_train_scaled, y_train)

# Prediction using SGDClassifier
sgd_prediction = sgd.predict(X_test_scaled)

# Evaluations
print('Classification Report: \n')
print(classification_report(y_test, sgd_prediction))
print('\nConfusion Matrix: \n')
print(confusion_matrix(y_test,sgd_prediction))

# Cross validation
print('\nCross validation: \n')
recall_estimate = cross_val_score(sgd, X_test_scaled, y_test, cv=10, scoring='recall_weighted')
print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

y_pred = cross_val_predict(sgd, X_test_scaled, y_test, cv=10)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)

# Tuning RF===============================================

print('\nTuning RF: \n')

# Re-instantiate and retrain the model to find feature importances
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train_scaled, y_train)

importances = pd.DataFrame({'Feature':stars.columns.drop('target_class'), 'Importance':rf.feature_importances_}).set_index('Feature')
importances = importances.sort_values('Importance', ascending=False)

print(importances)
importances.plot.bar()

plt.show()

# Dropping last 2 features
X_train_dropped = X_train_scaled.drop(list(importances.index[-2:]), axis=1)
X_test_dropped = X_test_scaled.drop(list(importances.index[-2:]), axis=1)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_model.fit(X_train_scaled, y_train)
print("Model's accuracy before dropping weak features: ", rf_model.score(X_test_scaled, y_test))
print("Model's accuracy after dropping weak features: ", rf.score(X_test, y_test))

# Tune the model
rf = RandomForestClassifier(n_estimators=120)
rf.fit(X_train_dropped, y_train)

# 180-80
# 120-62
# 100-69
# 110-76
# 130-87

# Cross validation
print('\nCross validation of tunned RF: \n')
recall_estimate = cross_val_score(rf, X_test_dropped, y_test, cv=10, scoring='recall_weighted')
print("Recall: " + str(round(100*recall_estimate.mean(), 2)) + "%")

y_pred = cross_val_predict(rf, X_test_dropped, y_test, cv=10)
conf_mat = confusion_matrix(y_test, y_pred)

print(conf_mat)








