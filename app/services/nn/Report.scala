// Neural Network - The Hard Way
// Wei Chen
// 2018-10-08
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

class Report {
    var iter = 0
    var lossTrain = 0.0
    var lossTest = 0.0
    def show: Unit = {
        println("Iteration  : " + iter)
        println("Loss Train : " + lossTrain)
        println("Loss Test  : " + lossTest)
    }
    override def toString: String = {
        s"""
{
    "iter": ${iter},
    "lossTrain": ${lossTrain},
    "lossTest": ${lossTest}
}
        """
    }
}