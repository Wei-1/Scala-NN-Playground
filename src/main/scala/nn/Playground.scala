// Neural Network - The Hard Way
// Wei Chen
// 2018-10-04
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

class State {
    var learningRate = 0.03
    var regularizationRate = 0
    var showTestData = false
    var noise = 0
    var batchSize = 10
    var discretize = false
    var tutorial: String = null
    var percTrainData = 50
    var activation: ActivationFunction = TANH
    var regularization: RegularizationFunction = null
    var problem = "CLASSIFICATION"
    var initZero = false
    var hideText = false
    var collectStats = false
    var numHiddenLayers = 1
    var hiddenLayerControls: Array[Any] = null
    var networkShape: Array[Int] = Array(4, 2)
    var x = true
    var y = true
    var xTimesY = false
    var xSquared = false
    var ySquared = false
    var cosX = false
    var sinX = false
    var cosY = false
    var sinY = false
    var dataset: (Int, Double) => Array[Example2D] = DataSet.classifyCircleData
    var regDataset: (Int, Double) => Array[Example2D] = DataSet.regressPlane
    var seed: String = ""
    var inputFormats: Array[String] = Array("x", "y", "x^2", "y^2", "x*y")
}

class UI {
    var lossTrain = 0.0
    var lossTest = 0.0
    def show: Unit = {
        println("Loss Train : " + lossTrain)
        println("Loss Test  : " + lossTest)
    }
}

class Playground {
    val RECT_SIZE = 30
    val BIAS_SIZE = 5
    val NUM_SAMPLES_CLASSIFY = 500
    val NUM_SAMPLES_REGRESS = 1200
    val DENSITY = 100

    var iter = 0
    var network = Array[Array[Node]]()
    var trainData = Array[Example2D]()
    var testData = Array[Example2D]()
    var state = new State
    var ui = new UI

    val inputDataMap = Map(
        "x" -> ((x: Double, y: Double) => (x)),
        "y" -> ((x: Double, y: Double) => (y)),
        "x^2" -> ((x: Double, y: Double) => (x * x)),
        "y^2" -> ((x: Double, y: Double) => (y * y)),
        "x*y" -> ((x: Double, y: Double) => (x * y))
    )

    def constructInput(x: Double, y: Double): Array[Double] = {
        var input = Array[Double]()
        for (inputName <- state.inputFormats) {
            if (inputDataMap.contains(inputName)) {
                input :+= inputDataMap(inputName)(x, y)
            }
        }
        input
    }

    def oneStep(): Unit = {
        iter += 1
        for(i <- trainData.indices) {
            val point = trainData(i)
            val input = constructInput(point.x, point.y)
            NeuralNetwork.forwardProp(network, input)
            NeuralNetwork.backProp(network, point.label, SQUARE)
            if((i + 1) % state.batchSize == 0) {
                NeuralNetwork.updateWeights(network, state.learningRate, state.regularizationRate)
            }
        }
        // Compute the loss.
        ui.lossTrain = getLoss(network, trainData)
        ui.lossTest = getLoss(network, testData)
        updateUI()
    }

    def getLoss(network: Array[Array[Node]], dataPoints: Array[Example2D]): Double = {
        var loss = 0.0
        for(dataPoint <- dataPoints) {
            val input = constructInput(dataPoint.x, dataPoint.y)
            val output = NeuralNetwork.forwardProp(network, input)
            loss += SQUARE.error(output, dataPoint.label)
        }
        loss / dataPoints.length
    }

    def reset(onStartup: Boolean = false): Unit = {
        // Make a simple network.
        iter = 0
        val numInputs = constructInput(0 , 0).length
        val shape = numInputs +: state.networkShape :+ 1
        val outputActivation = if(state.problem == "REGRESSION") LINEAR else TANH
        network = NeuralNetwork.buildNetwork(shape, state.activation, outputActivation,
            state.regularization, state.inputFormats, state.initZero)
        ui.lossTrain = getLoss(network, trainData)
        ui.lossTest = getLoss(network, testData)
        updateUI(true)
    }

    def generateData(firstTime: Boolean = false): Unit = {
        val numSamples = if(state.problem == "REGRESSION") NUM_SAMPLES_REGRESS else NUM_SAMPLES_CLASSIFY
        val generator = if(state.problem == "CLASSIFICATION") state.dataset else state.regDataset
        val data = generator(numSamples, state.noise / 100)
        // Shuffle the data in-place.
        DataSet.shuffle(data)
        // Split into train and test data.
        val splitIndex = (data.length * state.percTrainData / 100).toInt
        trainData = data.take(splitIndex)
        testData = data.drop(splitIndex)
    }

    def updateUI(firstStep: Boolean = false): Unit = {
    }
}