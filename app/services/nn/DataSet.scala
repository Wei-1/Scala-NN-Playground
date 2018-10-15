// Neural Network - The Hard Way
// Wei Chen
// 2018-10-02
// From Google's NN-Playground in TypeScript to Scala
package com.interplanetarytech.nn

/**
 * A two dimensional example: x and y coordinates with the label.
 */
class Example2D(val x: Double, val y: Double, val label: Double) {}
class Point(val x: Double, val y: Double) {}

object DataSet {
    /**
     * Shuffles the array using Fisher-Yates algorithm. Uses the seedrandom
     * library as the random generator.
     */
    def shuffle[T](array: Array[T]): Unit = {
        var counter = array.length
        var temp: T = array.head
        var index = 0
        // While there are elements in the array
        while(counter > 0) {
            // Pick a random index
            index = (Math.random() * counter).toInt
            // Decrease counter by 1
            counter -= 1
            // And swap the last element with it
            temp = array(counter)
            array(counter) = array(index)
            array(index) = temp
        }
    }

    def classifyTwoGaussData(numSamples: Int, noise: Double): Array[Example2D] = {
        var points = Array[Example2D]()

        def varianceScale(x: Double): Double = {
            if(x > 0.5) 4
            else if(x < 0) 0.5
            else x * 7 + 0.5
        }

        val variance = varianceScale(noise)

        def genGauss(cx: Double, cy: Double, label: Double) {
            for(i <- 0 until numSamples / 2) {
                val x = normalRandom(cx, variance)
                val y = normalRandom(cy, variance)
                points :+= new Example2D(x, y, label)
            }
        }

        genGauss(2, 2, 1) // Gaussian with positive examples.
        genGauss(-2, -2, -1) // Gaussian with negative examples.
        points
    }

    def regressPlane(numSamples: Int, noise: Double): Array[Example2D] = {
        val radius = 6

        def labelScale(x: Double): Double = {
            if(x > 10) 1
            else if(x < -10) -1
            else x / 10
        }
        val getLabel = (x: Double, y: Double) => labelScale(x + y)

        var points = Array[Example2D]()
        for(i <- 0 until numSamples) {
            val x = randUniform(-radius, radius)
            val y = randUniform(-radius, radius)
            val noiseX = randUniform(-radius, radius) * noise
            val noiseY = randUniform(-radius, radius) * noise
            val label = getLabel(x + noiseX, y + noiseY)
            points :+= new Example2D(x, y, label)
        }
        points
    }

    def regressGaussian(numSamples: Int, noise: Double): Array[Example2D] = {
        var points = Array[Example2D]()

        def labelScale(x: Double): Double = {
            if(x > 2) 0
            else if(x < 0) 1
            else 1 - x / 2
        }

        val gaussians = Array(
            (-4, 2.5, 1),
            (0, 2.5, -1),
            (4, 2.5, 1),
            (-4, -2.5, -1),
            (0, -2.5, 1),
            (4, -2.5, -1)
        )

        def getLabel(x: Double, y: Double): Double = {
            // Choose the one that is maximum in abs value.
            var label: Double = 0
            val pointxy = new Point(x, y)
            gaussians.foreach { case (cx, cy, sign) =>
                val newLabel = sign * labelScale(dist(pointxy, new Point(cx, cy)))
                if(Math.abs(newLabel) > Math.abs(label)) {
                    label = newLabel
                }
            }
            label
        }
        val radius = 6
        for (i <- 0 until numSamples) {
            val x = randUniform(-radius, radius)
            val y = randUniform(-radius, radius)
            val noiseX = randUniform(-radius, radius) * noise
            val noiseY = randUniform(-radius, radius) * noise
            val label = getLabel(x + noiseX, y + noiseY)
            points :+= new Example2D(x, y, label)
        }
        points
    }

    def classifySpiralData(numSamples: Int, noise: Double): Array[Example2D] = {
        var points = Array[Example2D]()
        val n = numSamples / 2

        def genSpiral(deltaT: Double, label: Double) {
            for (i <- 0 until n) {
                val r = i / n * 5
                val t = 1.75 * i / n * 2 * Math.PI + deltaT
                val x = r * Math.sin(t) + randUniform(-1, 1) * noise
                val y = r * Math.cos(t) + randUniform(-1, 1) * noise
                points :+= new Example2D(x, y, label)
            }
        }

        genSpiral(0, 1) // Positive examples.
        genSpiral(Math.PI, -1) // Negative examples.
        points
    }

    def classifyCircleData(numSamples: Int, noise: Double): Array[Example2D] = {
        var points = Array[Example2D]()
        val radius = 5
        def getCircleLabel(p: Point, center: Point): Double = if(dist(p, center) < (radius * 0.5)) 1 else -1

        val thecenter = new Point(0, 0)
        // Generate positive points inside the circle.
        for(i <- 0 until numSamples / 2) {
            val r = randUniform(0, radius * 0.5)
            val angle = randUniform(0, 2 * Math.PI)
            val x = r * Math.sin(angle)
            val y = r * Math.cos(angle)
            val noiseX = randUniform(-radius, radius) * noise
            val noiseY = randUniform(-radius, radius) * noise
            val label = getCircleLabel(new Point(x + noiseX, y + noiseY), thecenter)
            points :+= new Example2D(x, y, label)
        }

        // Generate negative points outside the circle.
        for(i <- 0 until numSamples / 2) {
            val r = randUniform(radius * 0.7, radius)
            val angle = randUniform(0, 2 * Math.PI)
            val x = r * Math.sin(angle)
            val y = r * Math.cos(angle)
            val noiseX = randUniform(-radius, radius) * noise
            val noiseY = randUniform(-radius, radius) * noise
            val label = getCircleLabel(new Point(x + noiseX, y + noiseY), thecenter)
            points :+= new Example2D(x, y, label)
        }
        points
    }

    def classifyXORData(numSamples: Int, noise: Double): Array[Example2D] = {
        def getXORLabel(p: Point): Double = if(p.x * p.y >= 0) 1 else -1

        val padding = 0.3
        var points = Array[Example2D]()
        for(i <- 0 until numSamples) yield {
            var x = randUniform(-5, 5)
            x += (if(x > 0) padding else -padding)  // Padding.
            var y = randUniform(-5, 5)
            y += (if(y > 0) padding else -padding)
            val noiseX = randUniform(-5, 5) * noise
            val noiseY = randUniform(-5, 5) * noise
            val label = getXORLabel(new Point(x + noiseX, y + noiseY))
            points :+= new Example2D(x, y, label)
        }
        points
    }

    /**
     * Returns a sample from a uniform [a, b] distribution.
     * Uses the seedrandom library as the random generator.
     */
    def randUniform(a: Double, b: Double) = Math.random() * (b - a) + a

    /**
     * Samples from a normal distribution. Uses the seedrandom library as the
     * random generator.
     *
     * @param mean The mean. Default is 0.
     * @param variance The variance. Default is 1.
     */
    def normalRandom(mean: Double = 0, variance: Double = 1): Double = {
        var v1: Double = 0
        var v2: Double = 0
        var s: Double = 0
        do {
            v1 = 2 * Math.random() - 1
            v2 = 2 * Math.random() - 1
            s = v1 * v1 + v2 * v2
        } while(s > 1)

        val result = Math.sqrt(-2 * Math.log(s) / s) * v1
        mean + Math.sqrt(variance) * result
    }

    /** Returns the eucledian distance between two points in space. */
    def dist(a: Point, b: Point): Double = {
        val dx = a.x - b.x
        val dy = a.y - b.y
        Math.sqrt(dx * dx + dy * dy)
    }
}