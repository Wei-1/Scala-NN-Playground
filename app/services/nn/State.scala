// Neural Network - The Hard Way
// Wei Chen
// 2018-10-08
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

class State {
    var learningRate = 0.03
    var regularizationRate = 0
    var showTestData = false
    var noise = 0
    var batchSize = 10
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
    var dataset: (Int, Double) => Array[Example2D] = DataSet.classifyCircleData
    var regDataset: (Int, Double) => Array[Example2D] = DataSet.regressPlane
    var seed: String = ""
    var inputFormats: Array[String] = Array("x", "y", "x^2", "y^2", "x*y")

    override def toString: String = {
        s"""
{
    "learningRate": ${learningRate},
    "regularizationRate": ${regularizationRate},
    "showTestData": ${showTestData},
    "noise": ${noise},
    "batchSize": ${batchSize},
    "tutorial": "${tutorial}",
    "percTrainData": ${percTrainData},
    "activation": "${if(activation != null)activation.getClass.getName else "null"}",
    "regularization": "${if(regularization != null)regularization.getClass.getName else "null"}",
    "problem": "${problem}",
    "initZero": ${initZero},
    "hideText": ${hideText},
    "collectStats": ${collectStats},
    "numHiddenLayers": ${numHiddenLayers},
    "hiddenLayerControls": "Not Available Yet!",
    "networkShape": [${networkShape.mkString(",")}],
    "dataset": "${if(dataset != null)dataset.getClass.getName else "null"}",
    "regDataset": "${if(regDataset != null)regDataset.getClass.getName else "null"}",
    "seed": "${seed}",
    "inputFormats": [${inputFormats.mkString("\"", "\",\"", "\"")}]
}
        """
    }
}
