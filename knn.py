from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def knn(X_train, X_test, y_train, y_test):
    accuracies = {1: 0, 3: 0, 5: 0, 7: 0, 9: 0, 21: 0, 99: 0, 199: 0}
    for num in [1, 3, 5, 7, 9, 21, 99, 199]:
        # Create a K Nearest Neighbors Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=num)
        knn_classifier.fit(X_train, y_train)  # train data
        y_pred = knn_classifier.predict(X_test)  # predict data

        # Calculate the accuracy of the predictions
        accuracies[num] = accuracy_score(y_test, y_pred)

    # Find the number of neighbors with the highest accuracy
    best = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best]

    # Re-run with the best number of neighbors
    knn_classifier = KNeighborsClassifier(n_neighbors=best)
    knn_classifier.fit(X_train, y_train)  # train data
    y_pred = knn_classifier.predict(X_test)  # predict data
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)  # Generate report of the values

    # Print the results
    print(f"Best k: {best} with Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return y_pred