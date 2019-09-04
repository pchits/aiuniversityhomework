import pandas as pd
import pickle
# Evaluations
from sklearn.metrics import classification_report, confusion_matrix
# Random Forest
from sklearn.ensemble import RandomForestClassifier
# Cross Val
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, cross_val_predict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Calling the function data_prep to get feature's columns
df_features = data_prep(stars)

# Spiting the data to train and test the model
X = df_features.copy()
y = stars['target_class'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Calling the scaler function by passing X_train and X_test to get the scaled data set
X_train_scaled = standardScaling(X_train)
X_test_scaled = standardScaling(X_test)

# During scaling process the index of X_train and X_test are changed.
# So, you must also reset the index of y_train and y_test as below, otherwise you will get index mismatch error.
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


X_test_scaled = X_test_scaled.drop("Excess_kurtosis_of_the_DM_SNR_curve", axis=1)

with open('pulsarmodel', 'rb') as f:
    rf_model = pickle.load(f)

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


# preds = rf.predict(new_X)