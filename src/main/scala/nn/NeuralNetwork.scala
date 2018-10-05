// Neural Network - The Hard Way
// Wei Chen
// 2018-10-02
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

/**
 * A node in a neural network. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
class Node(
    /**
     * Creates a new node with the provided id and activation function.
     */
    val id: String = null,
    /** Activation function that takes total input and returns node's output */
    val activation: ActivationFunction = null,
    val initZero: Boolean = false
) {
    /** List of input links. */
    var inputLinks = Array[Link]()
    var bias: Double = if(initZero) 0.0 else 0.1
    /** List of output links. */
    var outputs = Array[Link]()
    /** Node input and output. */
    var totalInput: Double = 0.0
    var output: Double = 0.0
    /** Error derivative with respect to this node's output. */
    var outputDer: Double = 0
    /** Error derivative with respect to this node's total input. */
    var inputDer: Double = 0
    /**
     * Accumulated error derivative with respect to this node's total input since
     * the last update. This derivative equals dE/db where b is the node's
     * bias term.
     */
    var accInputDer: Double = 0
    /**
     * Number of accumulated err. derivatives with respect to the total input
     * since the last update.
     */
    var numAccumulatedDers: Double = 0

    /** Recomputes the node's output and returns it. */
    def updateOutput(): Double = {
        // Stores total input into the node.
        totalInput = bias + inputLinks.map(link => link.weight * link.source.output).sum
        output = activation.output(totalInput)
        output
    }
}

/** Built-in error functions */
trait ErrorFunction {
    def error(output: Double, target: Double): Double
    def der(output: Double, target: Double): Double
}
object SQUARE extends ErrorFunction { // Square error only
    override def error(output: Double, target: Double): Double =
        0.5 * Math.pow(output - target, 2)
    override def der(output: Double, target: Double): Double =
        output - target
}

/** Built-in activation functions */
trait ActivationFunction {
    def output(x: Double): Double
    def der(x: Double): Double
}
object TANH extends ActivationFunction {
    override def output(x: Double): Double = Math.tanh(x)
    override def der(x: Double): Double = 1 - Math.pow(output(x), 2)
}
object RELU extends ActivationFunction {
    override def output(x: Double): Double = Math.max(0, x)
    override def der(x: Double): Double = if(x <= 0) 0 else 1
}
object SIGMOID extends ActivationFunction {
    override def output(x: Double): Double = 1 / (1 + Math.exp(-x))
    override def der(x: Double): Double = {
        val out = output(x)
        out * (1 - out)
    }
}
object LINEAR extends ActivationFunction {
    override def output(x: Double): Double = x
    override def der(x: Double): Double = 1
}

/** Build-in regularization functions */
trait RegularizationFunction {
    def output(w: Double): Double
    def der(w: Double): Double
}
object L1 extends RegularizationFunction {
    override def output(w: Double): Double = Math.abs(w)
    override def der(w: Double): Double = if(w < 0) -1 else if(w > 0) 1 else 0
}
object L2 extends RegularizationFunction {
    override def output(w: Double): Double = 0.5 * Math.pow(w, 2)
    override def der(w: Double): Double = w
}

/**
 * A link in a neural network. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
class Link(
    /**
    * Constructs a link in the neural network initialized with random weight.
    *
    * @param source The source node.
    * @param dest The destination node.
    * @param regularization The regularization function that computes the
    *     penalty for this weight. If null, there will be no regularization.
    */
    val source: Node,
    val dest: Node,
    val regularization: RegularizationFunction = null,
    val initZero: Boolean = false
) {
    val id: String = source.id + "-" + dest.id
    var weight: Double = if(initZero) 0 else Math.random() - 0.5;
    var isDead: Boolean = false
    /** Error derivative with respect to this weight. */
    var errorDer: Double = 0
    /** Accumulated error derivative since the last update. */
    var accErrorDer: Double = 0
    /** Number of accumulated derivatives since the last update. */
    var numAccumulatedDers: Double = 0
}

/**
 * A wrapper for all neural network functions
 */
object NeuralNetwork {
    /**
     * Builds a neural network.
     *
     * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
     *   the network will have one input node, 2 nodes in first hidden layer,
     *   3 nodes in second hidden layer and 1 output node.
     * @param activation The activation function of every hidden node.
     * @param outputActivation The activation function for the output nodes.
     * @param regularization The regularization function that computes a penalty
     *     for a given weight (parameter) in the network. If null, there will be
     *     no regularization.
     * @param inputIds List of ids for the input nodes.
     */
    def buildNetwork(
        networkShape: Array[Int],
        activation: ActivationFunction,
        outputActivation: ActivationFunction,
        regularization: RegularizationFunction,
        inputIds: Array[String],
        initZero: Boolean
    ): Array[Array[Node]] = {
        val numLayers = networkShape.length
        var id = 1
        /** List of layers, with each layer being a list of nodes. */
        var network = Array[Array[Node]]()
        for(layerIdx <- 0 until numLayers) {
            val isOutputLayer: Boolean = layerIdx == (numLayers - 1)
            val isInputLayer: Boolean = layerIdx == 0
            val numNodes = networkShape(layerIdx)
            val currentLayer = (for(i <- 0 until numNodes) yield {
                var nodeId = id.toString()
                if(isInputLayer) {
                    nodeId = inputIds(i)
                } else {
                    id += 1
                }
                val node = new Node(
                    nodeId,
                    if(isOutputLayer) outputActivation else activation,
                    initZero
                )
                if(layerIdx >= 1) {
                    // Add links from nodes in the previous layer to this node.
                    network(layerIdx - 1).foreach { prevNode =>
                        val link = new Link(prevNode, node, regularization, initZero)
                        prevNode.outputs :+= link
                        node.inputLinks :+= link
                    }
                }
                node
            }).toArray
            network :+= currentLayer
        }
        network
    }
    /**
     * Runs a forward propagation of the provided input through the provided
     * network. This method modifies the internal state of the network - the
     * total input and output of each node in the network.
     *
     * @param network The neural network.
     * @param inputs The input array. Its length should match the number of input
     *     nodes in the network.
     * @return The final output of the network.
     */
    def forwardProp(
        network: Array[Array[Node]],
        inputs: Array[Double]
    ): Double = {
        val inputLayer = network.head
        if(inputs.length != inputLayer.length) {
            Console.err.println("The number of inputs must match the number of nodes in the input layer")
            System.exit(1)
        }
        // Update the input layer.
        for(i <- 0 until inputLayer.length) {
            val node = inputLayer(i)
            node.output = inputs(i)
        }
        network.drop(1).foreach { currentLayer =>
            // Update all the nodes in this layer.
            for(node <- currentLayer) node.updateOutput()
        }
        network.last.head.output
    }

    /**
     * Runs a backward propagation using the provided target and the
     * computed output of the previous call to forward propagation.
     * This method modifies the internal state of the network - the error
     * derivatives with respect to each node, and each weight
     * in the network.
     */
    def backProp(
        network: Array[Array[Node]],
        target: Double,
        errorFunc: ErrorFunction
    ): Unit = {
        // The output node is a special case. We use the user-defined error
        // function for the derivative.
        val outputNode = network.last.head
        outputNode.outputDer = errorFunc.der(outputNode.output, target)

        // Go through the layers backwards.
        for(layerIdx <- network.length - 1 to 1 by -1) {
            val currentLayer = network(layerIdx)
            // Compute the error derivative of each node with respect to:
            // 1) its total input
            // 2) each of its input weights.
            for(node <- currentLayer) {
                node.inputDer = node.outputDer * node.activation.der(node.totalInput)
                node.accInputDer += node.inputDer
                node.numAccumulatedDers += 1
            }

            // Error derivative with respect to each weight coming into the node.
            for(node <- currentLayer) {
                for(link <- node.inputLinks) {
                    if(!link.isDead) {
                        link.errorDer = node.inputDer * link.source.output
                        link.accErrorDer += link.errorDer
                        link.numAccumulatedDers += 1
                    }
                }
            }
            if(layerIdx > 1) {
                val prevLayer = network(layerIdx - 1)
                for(node <- prevLayer) {
                    // Compute the error derivative with respect to each node's output.
                    node.outputDer = 0;
                    for(output <- node.outputs) {
                        node.outputDer += output.weight * output.dest.inputDer
                    }
                }
            }
        }
    }

    /**
     * Updates the weights of the network using the previously accumulated error
     * derivatives.
     */
    def updateWeights(
        network: Array[Array[Node]],
        learningRate: Double,
        regularizationRate: Double
    ): Unit = {
        for(currentLayer <- network.drop(1); node <- currentLayer) {
            // Update the node's bias.
            if(node.numAccumulatedDers > 0) {
                node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers
                node.accInputDer = 0
                node.numAccumulatedDers = 0
            }
            // Update the weights coming into this node.
            for(link <- node.inputLinks) {
                if(!link.isDead) {
                    if(link.numAccumulatedDers > 0) {
                        // Update the weight based on dE/dw.
                        link.weight = link.weight -
                            (learningRate / link.numAccumulatedDers) * link.accErrorDer
                        // Further update the weight based on regularization.
                        val regulDer = if(link.regularization != null) link.regularization.der(link.weight) else 0
                        val newLinkWeight = link.weight - (learningRate * regularizationRate) * regulDer
                        if(link.regularization != null &&
                            link.regularization.der(2) == 1 && // regularization == L1
                            link.weight * newLinkWeight < 0) {
                            // The weight crossed 0 due to the regularization term. Set it to 0.
                            link.weight = 0
                            link.isDead = true
                        } else {
                            link.weight = newLinkWeight
                        }
                        link.accErrorDer = 0
                        link.numAccumulatedDers = 0
                    }
                }
            }
        }
    }

    /** Iterates over every node in the network/ */
    def forEachNode(
        network: Array[Array[Node]],
        ignoreInputs: Boolean,
        accessor: Node => Unit
    ): Unit = {
        if(ignoreInputs) network.drop(1)
        else network
    }.foreach { currentLayer =>
        currentLayer.foreach(node => accessor(node))
    }

    /** Returns the output node in the network. */
    def getOutputNode(network: Array[Array[Node]]): Node =
        network.last.head;
}