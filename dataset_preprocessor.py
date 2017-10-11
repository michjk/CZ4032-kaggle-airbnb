from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def categorize_label(Y):
    label_size = len(Y)
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(Y)
    label_encoded = label_encoder.transform(Y)

    return label_encoded

def feature_engineer_id(X):
    
