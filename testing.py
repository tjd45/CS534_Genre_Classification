from knn import knn
from nb import nb
from sgd import sgd

from initialPreprocessing import gen_Train_and_Test, top_tracks, top_n_genre_tracks, top_echonest_tracks, top_tracks_final


def runTests(sample, featSelection):
    X_train, X_test, y_train, y_test = gen_Train_and_Test(sample,featSelection,0)

    print(f"\n\nTesting for {featSelection}")
    nb_acc, nb_pred = nb(X_train, X_test, y_train, y_test)
    sgd_acc, sgd_pred = sgd(X_train, X_test, y_train, y_test)
    knn_acc, knn_pred, knn_bestk = knn(X_train, X_test, y_train, y_test)

    resultString = ""
    resultString += f"RESULTS FOR {featSelection},{nb_acc:.2f},{sgd_acc:.2f},{knn_acc:.2f},{knn_bestk}\n"
    print(resultString)
    return resultString


def runSingleFeatures(datasetindex=0):
    tracks = top_tracks_final()

    fullResults = ""

    for i in range(4):
        fullResults += "DATASET "+str(i) +"\n"
        if i == 1 or i == 3:
            single_feats = ['track_duration','track_listens','track_favorites','days_since_first']
        else:
            single_feats = ['track_duration','track_listens','track_favorites']
    

        for feat in single_feats:
            fullResults+=runTests(tracks[i],feat)
        
        fullResults+="\n"


    
    print(fullResults)


runSingleFeatures()

