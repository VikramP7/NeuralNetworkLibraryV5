# Neural Network Library
A library for using and training feed forward neural networks of variable architecture written in java.

Started 2021 Completed 2024

![Image of Neural Network Diagram][imgLink]

## Brief Description
This project consists of two java files: the Matrix.java and the NeuralNetwork.java. The matrix object is a matrix library that handles all of the matrix mathematics *(as expected...)* and acts as a data structure to hold the values of the neural network. The neural network object is a library that allows the user to create **feed forward** networks of specified topology and train the network using sample data and the **gradient decent back propagation** model.

This project is a library written completely from scratch by me. I created this mostly as an educational experience, exploring how matrix mathematics works and how neural networks function.

If you want to see a working example of the library click [here][linkHandwritten].

![Image of Weight Matrix Data][imgSampleMatrix]



## Documentation
### Initialization

`NeuralNetwork(#ofInputs, {#ofNodesInFirstHiddenLayer, #ofNodesInSecondHiddenLayer,...}, #ofOutputs, [randomSeed])`

There are many constructors to meet various needs, however most constructors require the topology of the network. How many inputs will the network take, how many hidden layers will there be (and how many nodes should each one have), and how many output will the network produce. 

The library is also equipped to create a network based on a JSON file earlier exported from the program. *When I said this library was written entirely from scratch, this JSON interpreter was too... It hasn't been as extensively tested*

### Getting and Setting Network Parameters
`SetLearningRate(float learningRate)`: Get and Set the value of the learning rate. 

`GetStructure()`: Returns an integer array containing the number of nodes at each layer. 

`GetWeights() or GetBiases()`: Returns an array of the weight or bias matrices.

### Using the Neural Network
`FeedForward(float[] inputArray, [PrintOption])`: When provided an array of input values it returns an array of the output values of the matrix. See docstring for print options.

### Training the Neural Network
`Train(float[][] inputData, float[][] targetsData, int batchSize, int iterations, printOptions)`: Provide all of the training inputs and targets in the inputData and targetsData. Set the batch size and how many iterations  over the provided training data should be completed. And provide the print options. This will train the neural network.

`Test(float[][] testInputs, float[][] testTargets, printProgress, resultType, resultPrintOptions)`: Provide this function with your testing data and the type of output (result) you expect and it will create an accuracy report of the neural network.

 :warning: *This is not very good documentation. The docstrings in the NeuralNetwork.java provide much better explainations.* :warning:

[imgLink]: /img/Neural-Networks-Architecture.png
[imgSampleMatrix]: /img//sample_matrix.png
[linkHandwritten]: https://github.com/VikramP7/HandWrittenNumberIdentifierNeuralNetwork