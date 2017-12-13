import numpy as np
import artificial_neural_network as ann


if __name__ == "__main__":
    # input dataset
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    # output dataset
    y = np.array([[0], [0], [1], [1]])
    clf = ann.NeuralNetworkClassifer(NeuronsNum=5, FeatureLen=X.shape[1], Learn_rate=0.5)
    for _ in xrange(100):
        clf.fit(X, y)

    print clf.predict(np.array([[0,1,0]]))