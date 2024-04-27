from initialPreprocessing import gen_Train_and_Test, top_tracks, top_n_genre_tracks, top_echonest_tracks
import pandas as pd
from svm import svm
from knn import knn
from nb import nb
from sgd import sgd

if __name__ == "__main__":                                                       
    # sample = top_tracks()
    sample = top_n_genre_tracks(7)

    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,'track_duration',0)

    print("\n\nTesting for Single Feature - duration")
    nb(X_train, X_test, y_train, y_test)
    knn(X_train, X_test, y_train, y_test)
    sgd(X_train, X_test, y_train, y_test)


