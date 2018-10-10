// Neural Network - The Hard Way
// Wei Chen
// 2018-10-08
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

class Report {
    var lossTrain = 0.0
    var lossTest = 0.0
    def show: Unit = {
        println("Loss Train : " + lossTrain)
        println("Loss Test  : " + lossTest)
    }
}