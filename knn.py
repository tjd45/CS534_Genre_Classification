from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def knn(X_train, X_test, y_train, y_test, desc="", N=5):

    print("\n\n\nKNN"+"("+str(N)+")"+desc+":\n")

    # Create a K Nearest Neighbors Classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=N)
    knn_classifier.fit(X_train, y_train) # train data
    y_pred = knn_classifier.predict(X_test) # predict data

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Generate report of the values

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return y_pred