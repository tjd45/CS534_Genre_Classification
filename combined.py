from initialPreprocessing import gen_Train_and_Test, top_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb

if __name__ == "__main__":                                                       
    sample = top_tracks()
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'',0,None,['track_duration','track_listens','track_favorites'])

    print("\n\nTesting for Multi Features")
    nb(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test,"",9)
    knn(X_train, X_test, y_train, y_test,"",21)
    knn(X_train, X_test, y_train, y_test,"",99)
    knn(X_train, X_test, y_train, y_test,"",199)
