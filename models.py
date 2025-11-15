from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from enum import Enum

class model_types(Enum):
    Normal_model = 1
    Synthetic_model = 2



def load_logisitc_regression(X_train, X_test, y_train, y_test, model_type, class_label='class', train_model=True):
    # One-hot encode categorical features
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[categorical_features],)
    X_train_encoded.columns = encoder.get_feature_names_out(categorical_features)
    X_test_encoded = encoder.transform(X_test[categorical_features])
    X_test_encoded.columns = encoder.get_feature_names_out(categorical_features)

    # Combine encoded features with numerical features
    numerical_features = X_train.select_dtypes(include=['number']).columns
    X_train_numerical = X_train[numerical_features]
    X_test_numerical = X_test[numerical_features]

    X_train_final = pd.concat([pd.DataFrame(X_train_encoded.toarray(),
                                            columns=encoder.get_feature_names_out(categorical_features),
                                            index=X_train_numerical.index),
                               X_train_numerical], axis=1)
    X_test_final = pd.concat([pd.DataFrame(X_test_encoded.toarray(),
                                           columns=encoder.get_feature_names_out(categorical_features),
                                           index=X_test_numerical.index),
                              X_test_numerical], axis=1)
    model = None
    if train_model:
        # Train a Logistic Regression model
        model = LogisticRegression(random_state=13)
        model.fit(X_train_final, y_train)
    if model_type == model_types.Normal_model and train_model:

        # Predict labels for the test data
        y_pred = model.predict(X_test_final)

        # Replace the labels in the test data with the predicted labels
        X_test[class_label] = y_pred
        X_test_final[class_label] = y_pred

    else:
        X_test[class_label] = y_test
        X_test_final[class_label] = y_test

    return model, X_test, X_test_final, encoder
