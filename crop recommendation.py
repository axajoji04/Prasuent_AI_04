import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the data
crop = pd.read_csv("Crop_recommendation.csv")

# Data preprocessing
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)

# Feature and target split
X = crop.drop(['crop_num', 'label'], axis=1)
y = crop['crop_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
ms = MinMaxScaler()
sc = StandardScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

import joblib

# Save the model and scalers using joblib
joblib.dump(rfc, 'model.pkl')
joblib.dump(ms, 'minmaxscaler.pkl')
joblib.dump(sc, 'standscaler.pkl')
