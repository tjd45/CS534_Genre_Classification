from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

import numpy as np

# default is now for it to search for best value of k, if you want to change this behaviour then need to change N=-1 to N= chosen number of k neighbours
def knn(X_train, X_test, y_train, y_test, desc="", N=-1):

    bestN = -1

    # Think this is basically a reimplementation of what you did before, but this way you can specify if you want to seek the best k
    # This is basically a like nested CV lite so we definitely should investigate doing that for any final implementation we do
    if N<0:
        sqrt_sample_size = int(np.sqrt(X_train.shape[0]))

        # Generate odd k values from 1 to sqrt_sample_size
        odd_k_values = list(range(1, sqrt_sample_size + 1, 2))

        param_grid = {'n_neighbors': odd_k_values}

        knn_classifier = KNeighborsClassifier()
    
        # Perform GridSearchCV
        grid_search = GridSearchCV(knn_classifier, param_grid, cv=5)  # 5-fold cross-validation
        grid_search.fit(X_train, y_train)  # Fit the model with training data

        # Best hyperparameters
        best_params = grid_search.best_params_
        best_k = best_params['n_neighbors']

        # Train the model with the best k value
        best_knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
        best_knn_classifier.fit(X_train, y_train)  # Train data
        y_pred = best_knn_classifier.predict(X_test)  # Predict data

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)  # Generate report

        # Print the results
        print(f"Best k: {best_k}")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(report)

        return accuracy, y_pred, best_k

        
    num_feats = N

    if bestN>0:
        num_feats = bestN


    print("\n\n\nKNN"+"("+str(num_feats)+")"+desc+":\n")
    print(str(X_train.shape[1]) + " features")

    # Create a K Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=num_feats)
    knn_classifier.fit(X_train, y_train) # train data
    y_pred = knn_classifier.predict(X_test) # predict data

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Generate report of the values

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return accuracy,y_pred, num_feats