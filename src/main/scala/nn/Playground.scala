// Neural Network - The Hard Way
// Wei Chen
// 2018-10-04
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

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
    var report = new Report

    val inputDataMap = Map(
        "x" -> ((x: Double, y: Double) => (x)),
        "y" -> ((x: Double, y: Double) => (y)),
        "x^2" -> ((x: Double, y: Double) => (x * x)),
        "y^2" -> ((x: Double, y: Double) => (y * y)),
        "x*y" -> ((x: Double, y: Double) => (x * y)),
        "cos_x" -> ((x: Double, y: Double) => (Math.cos(x))),
        "cos_y" -> ((x: Double, y: Double) => (Math.cos(y))),
        "sin_x" -> ((x: Double, y: Double) => (Math.sin(x))),
        "sin_y" -> ((x: Double, y: Double) => (Math.sin(y)))
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
        updateReport()
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
        updateReport(true)
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

    def updateReport(firstStep: Boolean = false): Unit = {
        // Compute the loss.
        report.lossTrain = getLoss(network, trainData)
        report.lossTest = getLoss(network, testData)
    }
}