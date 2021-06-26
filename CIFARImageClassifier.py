import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def LoadBatch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        X = data[b"data"]
        X = X.T

        y = data[b"labels"]
        Y = np.eye(10)[y]

        # Reformat
        Y = Y.T

    return X, Y, y


def LoadAllData(path, NmbTrainImages=45000):

    dirs = os.listdir(path)

    dirs = [x for x in dirs if "data" in x and not "1" in x]
    dirs.sort()

    X, Y, y = \
        LoadBatch("../../Datasets/cifar-10-batches-py/data_batch_1")

    for file in dirs:
        filePath = str(path + file)
        XNew, YNew, yNew = LoadBatch(filePath)
        X = np.concatenate((X, XNew), axis=1)
        Y = np.concatenate((Y, YNew), axis=1)
        y = np.concatenate((y, yNew))

    trainData, validationData, testData = dict(), dict(), dict()

    trainData['X'] = X[:, :NmbTrainImages]
    trainData['Y'] = Y[:, :NmbTrainImages]
    trainData['y'] = y[:NmbTrainImages]

    validationData['X'] = X[:, NmbTrainImages:]
    validationData['Y'] = Y[:, NmbTrainImages:]
    validationData['y'] = y[NmbTrainImages:]

    Xtest, Ytest, ytest = LoadBatch(path + 'test_batch')
    testData['X'] = Xtest
    testData['Y'] = Ytest
    testData['y'] = ytest

    XTrainMean = np.mean(trainData['X'], axis=1, keepdims=True)
    XTrainStd = np.std(trainData['X'], axis=1, keepdims=True)

    trainData['X'] = (trainData['X'] - XTrainMean) / XTrainStd
    validationData['X'] = (validationData['X'] - XTrainMean) / XTrainStd
    testData['X'] = (testData['X'] - XTrainMean) / XTrainStd

    return trainData, validationData, testData


class NeuralNetwork():

    def __init__(self, labels, layers, alpha=0.5, batchNormalise=False, xavier=True, sig=1e-1):

        self.labels = labels

        self.nLayers = len(layers) - 1
        self.alpha = alpha
        self.batchNormalise = batchNormalise

        self.activationFuncs = {
            'softmax': self.ComputeSoftmax, 'relu': self.ComputeRelu}

        self.W = []
        self.b = []
        self.whichActFunc = []
        self.gamma = []
        self.beta = []
        self.varian = []
        self.muu = []

        self.sig = sig
        self.xavier = xavier

        if self.batchNormalise:
            self.parameters = {"W": self.W, "b": self.b, "gamma": self.gamma,
                               "beta": self.beta}
        else:
            self.parameters = {"W": self.W, "b": self.b}

        self.layers = layers
        prevShape = layers["layer0"]["shape"]

        for layer in layers.values():
            for typee, shape in layer.items():
                if typee == "shape":
                    W, b, gamma, beta, muu, varian = self.InitParameters(
                        shape, prevShape, xavier=self.xavier, sig=self.sig)
                    self.W.append(W), self.b.append(b)
                    self.gamma.append(gamma), self.beta.append(beta)
                    self.muu.append(muu), self.varian.append(varian)
                    prevShape = shape
                else:
                    self.whichActFunc.append(
                        (shape, self.activationFuncs[shape]))

    def InitParameters(self, layerShape, prevShape, xavier=False, sig=1e-1):

        gamma = np.ones((layerShape[0], 1))
        mu = np.zeros((layerShape[0], 1))
        beta = np.zeros((layerShape[0], 1))
        var = np.zeros((layerShape[0], 1))

        if xavier:
            W = np.random.normal(0, 1/np.sqrt(prevShape[0]),
                                 size=(layerShape[0], layerShape[1]))
        else:
            W = np.random.normal(0, sig, size=(layerShape[0], layerShape[1]))

        b = np.zeros(layerShape[0]).reshape(layerShape[0], 1)

        return W, b, gamma, beta, mu, var

    def ComputeSoftmax(self, x):
        s = np.exp(x - np.max(x, axis=0)) / \
            np.exp(x - np.max(x, axis=0)).sum(axis=0)

        return s

    def ComputeRelu(self, x):
        x[x < 0] = 0
        return x

    def EvaluateNetworkWithBN(self, X, testing=False, training=False):
        s = np.copy(X)
        S = []
        S2 = []
        means = []
        variances = []
        H = []

        for i, (W, b, gamma, beta, muu, varian, activation) in enumerate(
            zip(self.W, self.b, self.gamma, self.beta, self.muu,
                self.varian, self.whichActFunc)):

            H.append(s)
            s = W@s + b

            if i < self.nLayers:
                S.append(s)
                if testing:
                    varEps = np.sqrt(varian + np.finfo(np.float64).eps)
                    s = (s - muu) / varEps
                else:
                    variance = np.var(s, axis=1, keepdims=True) * \
                        (X.shape[1]-1)/X.shape[1]
                    variances.append(variance)
                    newMu = np.mean(s, axis=1, keepdims=True)
                    means.append(newMu)

                    if training:
                        self.varian[i] = varian * self.alpha + \
                            (1-self.alpha) * variance

                        self.muu[i] = muu * self.alpha + \
                            (1-self.alpha) * newMu

                    varEps = np.sqrt(variance +
                                     np.finfo(np.float64).eps)
                    s = (s - newMu) / varEps

                S2.append(s)
                sTm = np.multiply(gamma, s) + beta
                s = activation[1](sTm)

            else:
                P = activation[1](s)

        return H, P, S, S2, means, variances

    def EvaluateNetwork(self, X):
        s = np.copy(X)
        H = []
        for W, b, act in zip(self.W, self.b, self.whichActFunc):
            if act[0] == "relu":
                s = self.ComputeRelu(W@s + b)
                H.append(s)
            else:
                P = self.ComputeSoftmax(W@s + b)

        return H, P

    def ComputeCost(self, X, Y, lambdas, testing=False):

        sqW = 0

        if self.batchNormalise:
            H, P, S, S2, means, variances = self.EvaluateNetworkWithBN(
                X, testing=testing)
        else:
            H, P = self.EvaluateNetwork(X)

        loss = 1/X.shape[1] * - np.sum(Y*np.log(P))

        for W in self.W:
            sqW += (np.sum(np.square(W)))

        regTerm = lambdas * sqW
        cost = loss + regTerm

        return loss, cost

    def ComputeAccuracy(self, X, y, testing=False):
        if self.batchNormalise:
            predictions = self.EvaluateNetworkWithBN(X, testing=testing)
        else:
            predictions = self.EvaluateNetwork(X)

        bestPrediction = np.argmax(predictions[1], axis=0)
        totalAccuracy = bestPrediction.T[bestPrediction == np.asarray(
            y)].shape[0]

        return totalAccuracy / X.shape[1]

    def ComputeGradientsWithBN(self, XBatch, YBatch, lambdas):

        N = XBatch.shape[1]

        gradWs = []
        gradbs = []
        gradgammas = []
        gradbetas = []

        for Ws in self.parameters["W"]:
            gradWs.append(np.zeros_like(Ws))

        for bs in self.parameters["b"]:
            gradbs.append(np.zeros_like(bs))

        for gammas in self.parameters["gamma"]:
            gradgammas.append(np.zeros_like(gammas))

        for betas in self.parameters["beta"]:
            gradbetas.append(np.zeros_like(betas))

        Hbatch, Pbatch, Sbatch, S2batch, meanBatch, variationBatch = \
            self.EvaluateNetworkWithBN(XBatch, training=True)

        Gbatch = - (YBatch - Pbatch)

        gradWs[self.nLayers] = 1/N * Gbatch@Hbatch[self.nLayers].T + \
            2 * lambdas * self.W[self.nLayers]
        gradbs[self.nLayers] = np.reshape(1/N * Gbatch@np.ones(N),
                                          (gradbs[self.nLayers].shape[0], 1))

        Gbatch = self.W[self.nLayers].T@Gbatch
        Hbatch[self.nLayers][Hbatch[self.nLayers] <= 0] = 0
        Gbatch = np.multiply(Gbatch, Hbatch[self.nLayers] > 0)

        # Compute backwards
        for nLayer in range(self.nLayers-1, -1, -1):
            gradgammas[nLayer] = 1/N * \
                np.multiply(Gbatch, S2batch[nLayer])@np.ones(N)
            gradgammas[nLayer] = np.reshape(
                gradgammas[nLayer], (gradgammas[nLayer].shape[0], 1))

            gradbetas[nLayer] = 1/N * Gbatch@np.ones(N)
            gradbetas[nLayer] = np.reshape(
                gradbetas[nLayer], (gradbetas[nLayer].shape[0], 1))

            Gbatch = np.multiply(Gbatch, self.gamma[nLayer])
            Gbatch = self.BatchNormalise(Gbatch, Sbatch[nLayer],
                                         meanBatch[nLayer], variationBatch[nLayer])

            gradWs[nLayer] = 1/N * Gbatch@Hbatch[nLayer].T + \
                2 * lambdas * self.W[nLayer]

            gradbs[nLayer] = np.reshape(1/N * Gbatch@np.ones(N),
                                        (gradbs[nLayer].shape[0], 1))
            if not nLayer == 0:
                Gbatch = self.W[nLayer].T@Gbatch
                Hbatch[nLayer][Hbatch[nLayer] <= 0] = 0
                Gbatch = np.multiply(Gbatch, Hbatch[nLayer] > 0)

        return gradWs, gradbs, gradgammas, gradbetas

    def ComputeGradients(self, XBatch, YBatch, lambdas):

        gradWs, gradbs, gradgammas, gradbetas = [], [], None, None
        for W, b in zip(self.W, self.b):
            gradWs.append(np.zeros_like(W))
            gradbs.append(np.zeros_like(b))

        Hbatch, Pbatch = self.EvaluateNetwork(XBatch)
        Gbatch = - (YBatch - Pbatch)

        # Compute backwards
        for l in range(self.nLayers, 0, -1):
            gradWs[l] = 1/XBatch.shape[1] * \
                Gbatch@Hbatch[l-1].T + 2 * lambdas * self.W[l]
            gradbs[l] = np.reshape(
                1/XBatch.shape[1] * Gbatch@np.ones(XBatch.shape[1]), (gradbs[l].shape[0], 1))

            Gbatch = self.W[l].T@Gbatch
            Hbatch[l-1][Hbatch[l-1] <= 0] = 0
            Gbatch = np.multiply(Gbatch, Hbatch[l-1] > 0)

        gradWs[0] = 1/XBatch.shape[1] * Gbatch@XBatch.T + lambdas * self.W[0]
        gradbs[0] = np.reshape(
            1/XBatch.shape[1] * Gbatch@np.ones(XBatch.shape[1]), self.b[0].shape)

        return gradWs, gradbs, gradgammas, gradbetas

    def BatchNormalise(self, Gbatch, Sbatch, mean, variance):
        s1 = np.multiply(Gbatch, np.power(
            variance + np.finfo(np.float64).eps, -0.5))
        s2 = np.multiply(Gbatch, np.power(
            variance + np.finfo(np.float64).eps, -1.5))

        Gbatch = s1 - 1/Gbatch.shape[1] * np.sum(s1, axis=1, keepdims=True) - \
            1/Gbatch.shape[1] * np.multiply(Sbatch - mean,
                                            np.sum(np.multiply(s2, Sbatch - mean), axis=1, keepdims=True))

        return Gbatch

    def GenerateAccuracyGraphs(self, costsTraining, costsValidation, n_epochs, label="", yLimit=None, lambdas=None,
                               batchSize=None,
                               etaMin=None, etaMax=None,
                               nS=None, bN=False, sig=None):

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(np.arange(n_epochs), costsTraining, label="Training")
        ax.plot(np.arange(n_epochs), costsValidation, label="Validation")
        ax.legend()

        maxY = max(np.argmax(costsTraining), np.argmax(costsValidation))

        # if yLimit and (maxY < yLimit):
        #ax.set_ylim([0, yLimit])

        if bN:
            bN = "True"
        else:
            bN = "False"

        ax.set(xlabel='Epochs', ylabel=label)
        plt.figtext(0.5, 0.92, 'lambda:' + str(lambdas) + ', batchSize:' + str(batchSize) + ', etaMin:' +
                    str(etaMin) + ', etaMax:' + str(etaMax) + ', nS:' + str(nS) + ', bN:' + bN + ', sig:' + str(sig), ha='center',
                    bbox={"facecolor": "blue", "alpha": 0.2, "pad": 5})
        ax.grid()

        plt.savefig('output/' + label + 'Graph' + str(lambdas) + '_' +
                    str(etaMin) + '_' + str(etaMax) + '_' + str(nS) + '_bN' + bN + '_sig' + str(sig) + '.png')

    def MiniBatchGradientDescent(self, X, Y, lambdas=0, batchSize=100, etaMin=1e-5, etaMax=1e-1, nS=980, n_epochs=10, batchNormalise=False, generateGraphs=True, sig=1e-1):

        costsTraining = np.zeros(n_epochs)
        lossTraining = np.zeros(n_epochs)
        accuracyTraining = np.zeros(n_epochs)

        costsValidation = np.zeros(n_epochs)
        lossValidation = np.zeros(n_epochs)
        accuracyValidation = np.zeros(n_epochs)

        NmbOfBatches = int(np.floor(X.shape[1] / batchSize))
        eta = etaMin
        t = 0
        for NmbEpoch in range(n_epochs):
            print('Epoch: ' + str(NmbEpoch+1) + ' out of ' + str(n_epochs))
            for NmbBatch in range(NmbOfBatches):
                N = int(X.shape[1] / NmbOfBatches)
                jFrom = N * NmbBatch
                jTo = N * (NmbBatch+1)

                XBatch = X[:, jFrom:jTo]
                YBatch = Y[:, jFrom:jTo]

                if self.batchNormalise:
                    gradWs, gradbs, gradgammas, gradbetas = self.ComputeGradientsWithBN(
                        XBatch, YBatch, lambdas)
                else:
                    gradWs, gradbs, gradgammas, gradbetas = self.ComputeGradients(
                        XBatch, YBatch, lambdas)

                gradients = {"W": gradWs, "b": gradbs,
                             "gamma": gradgammas, "beta": gradbetas}

                for W, ch in zip(self.parameters["W"], gradients["W"]):
                    W -= eta * ch

                for b, ch in zip(self.parameters["b"], gradients["b"]):
                    b -= eta * ch

                if batchNormalise:
                    for gamma, ch in zip(self.parameters["gamma"], gradients["gamma"]):
                        gamma -= eta * ch

                    for beta, ch in zip(self.parameters["gamma"], gradients["gamma"]):
                        beta -= eta * ch

                if t <= nS:
                    eta = etaMin + t/nS * (etaMax - etaMin)

                elif t <= 2*nS:
                    eta = etaMax - (t - nS)/nS * (etaMax - etaMin)

                t = (t+1) % (2*nS)

            lossTraining[NmbEpoch], costsTraining[NmbEpoch] = self.ComputeCost(
                X, Y, lambdas)
            lossValidation[NmbEpoch], costsValidation[NmbEpoch] = self.ComputeCost(
                Xval, Yval, lambdas)

            accuracyTraining[NmbEpoch] = self.ComputeAccuracy(
                Xtrain, ytrain)
            print('Training Accuracy:', accuracyTraining[NmbEpoch])
            print('Training Loss:', lossTraining[NmbEpoch])
            print()
            accuracyValidation[NmbEpoch] = self.ComputeAccuracy(Xval, yval)

            # Shuffle training and labels in unison after each epoch
            from sklearn.utils import shuffle
            X, Y = shuffle(X.T, Y.T, random_state=0)
            X, Y = X.T, Y.T

        if generateGraphs:
            self.GenerateAccuracyGraphs(
                lossTraining, lossValidation, n_epochs, label="loss", yLimit=3,
                lambdas=lambdas,
                batchSize=batchSize,
                etaMin=etaMin, etaMax=etaMax,
                nS=nS, bN=batchNormalise, sig=sig
            )
            self.GenerateAccuracyGraphs(
                costsTraining, costsValidation, n_epochs, label="cost", yLimit=4,
                lambdas=lambdas,
                batchSize=batchSize,
                etaMin=etaMin, etaMax=etaMax,
                nS=nS, bN=batchNormalise, sig=sig
            )
            self.GenerateAccuracyGraphs(
                accuracyTraining, accuracyValidation, n_epochs, label="accuracy", yLimit=1,
                lambdas=lambdas,
                batchSize=batchSize,
                etaMin=etaMin, etaMax=etaMax,
                nS=nS, bN=batchNormalise, sig=sig
            )

        accuracyTraining = self.ComputeAccuracy(Xtrain, ytrain)
        accuracyValidation = self.ComputeAccuracy(Xval, yval)
        accuracyTest = self.ComputeAccuracy(Xtest, ytest,
                                            testing=True)

        print()
        print('Parameters:', 'lambdas=', lambdas, 'nS=',
              nS, 'etaMin=', etaMin, 'etaMax=', etaMax)
        print('Training Accuracy:', accuracyTraining)
        print('Validation Accuracy:', accuracyValidation)
        print('Test Accuracy:', accuracyTest)
        print()

        return accuracyTraining, accuracyValidation, accuracyTest


datasetPath = "../../Datasets/cifar-10-batches-py/"

# valSplit = 0.9: 10% of data is used for validation (5000 images)
# 0.98 = 1000 images
trainData, validationData, testData = LoadAllData(
    datasetPath, NmbTrainImages=45000)
# trainData, validationData, testData = LoadSmallerSample(datasetPath)


Xtrain, Ytrain, ytrain = trainData['X'], trainData['Y'], trainData['y']
Xval, Yval, yval = validationData['X'], validationData['Y'], validationData['y']
Xtest, Ytest, ytest = testData['X'], testData['Y'], testData['y']


labels_path = datasetPath + 'batches.meta'
with open(labels_path, 'rb') as f:
    labels = pickle.load(f, encoding='bytes')[b'label_names']


shapes, activations = [(50, 3072), (50, 50), (50, 50), (10, 50)], [
    "relu", "relu", "relu", "softmax"]

shapes2, activations2 = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20),
                         (10, 10), (10, 10), (10, 10), (10, 10)], [
    "relu", "relu", "relu", "relu", "relu", "relu",
    "relu", "relu", "softmax"]


layers = dict()
for i, (shape, activation) in enumerate(zip(shapes, activations)):
    lName = "layer" + str(i)
    layers[lName] = {"shape": shape, "activation": activation}

#lambdasCoarse = np.random.uniform(10**-5, 10**-1, 15)
#lambdasCoarse = np.random.uniform(.001, .004, 10)

# Hyperparameters
seeds = [1, 2, 3, 4, 5]
n_epochs = 3
batchSize = 100
NmbExperiments = 1
lambdas = [.003687, .003687, .003687, 0.01, .002852,  .004044]
etas = [(1e-5, 1e-1), (1e-5, 1e-1), (1e-5, 1e-1)]
nSs = [2250, 2450, 980, 1960, 2940]
xavier = True
sigs = [1e-1, 1e-1, 1e-3, 1e-3, 1e-4, 1e-4]


modelPerformance = {
    'training': [],
    'validation': [],
    'test': []
}

doBN = [True, False, False, True, False, True]

for i in range(0, 1):
    for n in range(NmbExperiments):
        if NmbExperiments > 1:
            print('Doing experiment ' + str(n+1) +
                  ' out of ' + str(NmbExperiments))
        np.random.seed(seeds[n])
        net = NeuralNetwork(layers=layers, labels=labels,
                            batchNormalise=doBN[i], xavier=xavier,
                            sig=sigs[i])
        trainingAccuracy, validationAccuracy, testAccuracy = net.MiniBatchGradientDescent(
            Xtrain[:, :10000], Ytrain[:, :10000],
            lambdas=lambdas[0],
            batchSize=batchSize,
            etaMin=etas[0][0], etaMax=etas[0][1],
            nS=nSs[0],
            n_epochs=n_epochs,
            batchNormalise=doBN[i],
            generateGraphs=True,
            sig=sigs[i])

        modelPerformance['training'].append(trainingAccuracy)
        modelPerformance['validation'].append(validationAccuracy)
        modelPerformance['test'].append(testAccuracy)


print(modelPerformance)
