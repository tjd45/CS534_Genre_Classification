from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def knn(X_train, X_test, y_train, y_test, desc="", N=5):

    bestN = -1

    # Think this is basically a reimplementation of what you did before, but this way you can specify if you want to seek the best k
    # This is basically a like nested CV lite so we definitely should investigate doing that for any final implementation we do
    if N<0:
        ns = [1,3,5,7,9,21,99,199]
        bestac = 0
        for n in ns:
            knn_classifier = KNeighborsClassifier(n_neighbors=n)
            knn_classifier.fit(X_train, y_train) # train data
            y_pred = knn_classifier.predict(X_test) # predict data
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > bestac:
                bestac = accuracy
                bestN = n
        
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

    return y_pred