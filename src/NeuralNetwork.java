import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class NeuralNetwork {
	private final int NumberOfInputNodes;
	private final int[] HiddenNodes;
	private final int NumberOfOutputNodes;
	private Matrix[] weightsMatricies;
	private Matrix[] biasMatricies;
	// For external speculation, no calculations are done to this or from this
	public Matrix[] neuronActivationMatricies;

	private float LEARNINGRATE = 8.5f;
	int BATCHSIZE = 1;

	// Sigmoid Activation Function
	public final Function<Float, Float> ACTIVATIONFUNCTION = x -> (1 / (1 + (float) Math.exp(-x)));
	public final Function<Float, Float> ACTIVATIONDERIVATIVEFUNCTION = x -> (float) ((Math.exp(-x))
			/ (Math.pow((1 + Math.exp(-x)), 2)));

	// ----------- CONSTRUCTORS -----------

	/**
	 * Constructor creates neural network given structure parameters and populates
	 * weights and biases using pseudo random seeded by curent time of execution
	 * 
	 * @param inputs  The number of input neurons in network
	 * 
	 * @param hidden  Array length specifying how many hidden layers and values
	 *                depicting how many neurons
	 * 
	 * @param outputs The number of output neurons in network
	 */
	public NeuralNetwork(int inputs, int[] hidden, int output) {
		this.NumberOfInputNodes = inputs;
		this.HiddenNodes = hidden;
		this.NumberOfOutputNodes = output;

		weightsMatricies = new Matrix[hidden.length + 1];
		weightsMatricies[0] = new Matrix(hidden[0], inputs);
		weightsMatricies[0].RandomFill();
		for (int i = 1; i < weightsMatricies.length - 1; i++) {
			weightsMatricies[i] = new Matrix(hidden[i], hidden[i - 1]);
			weightsMatricies[i].RandomFill();
		}
		weightsMatricies[weightsMatricies.length - 1] = new Matrix(output, hidden[hidden.length - 1]);
		weightsMatricies[weightsMatricies.length - 1].RandomFill();

		biasMatricies = new Matrix[hidden.length + 1];
		for (int i = 0; i < hidden.length; i++) {
			biasMatricies[i] = new Matrix(hidden[i], 1);
			biasMatricies[i].RandomFill();
		}
		biasMatricies[biasMatricies.length - 1] = new Matrix(output, 1);
		biasMatricies[biasMatricies.length - 1].RandomFill();

		neuronActivationMatricies = new Matrix[1 + hidden.length + 1]; // One for input layer, one for each hidden and
																		// one for ouput
		neuronActivationMatricies[0] = new Matrix(inputs, 1);
		for (int i = 0; i < hidden.length; i++) {
			neuronActivationMatricies[i + 1] = new Matrix(hidden[i], 1);
		}
		neuronActivationMatricies[neuronActivationMatricies.length - 1] = new Matrix(output, 1);
	}

	/**
	 * Constructor creates neural network given structure parameters and populates
	 * weights and biases using psuedo random seeded by provided seed
	 * 
	 * @param inputs  The number of input neurons in network
	 * 
	 * @param hidden  Array length specifying how many hidden layers and values
	 *                depicting how many neurons
	 * 
	 * @param outputs The number of output neurons in network
	 * 
	 * @param seed    The seed provided to the pseudo random initializer
	 */
	public NeuralNetwork(int inputs, int[] hidden, int output, long seed) {
		this.NumberOfInputNodes = inputs;
		this.HiddenNodes = hidden;
		this.NumberOfOutputNodes = output;
		Random random = new Random(seed);

		weightsMatricies = new Matrix[hidden.length + 1];
		weightsMatricies[0] = new Matrix(hidden[0], inputs);
		weightsMatricies[0].RandomFill(random.nextLong());
		for (int i = 1; i < weightsMatricies.length - 1; i++) {
			weightsMatricies[i] = new Matrix(hidden[i], hidden[i - 1]);
			weightsMatricies[i].RandomFill(random.nextLong());
		}
		weightsMatricies[weightsMatricies.length - 1] = new Matrix(output, hidden[hidden.length - 1]);
		weightsMatricies[weightsMatricies.length - 1].RandomFill(random.nextLong());

		biasMatricies = new Matrix[hidden.length + 1];
		for (int i = 0; i < hidden.length; i++) {
			biasMatricies[i] = new Matrix(hidden[i], 1);
			biasMatricies[i].RandomFill(random.nextLong());
		}
		biasMatricies[biasMatricies.length - 1] = new Matrix(output, 1);
		biasMatricies[biasMatricies.length - 1].RandomFill(random.nextLong());

		neuronActivationMatricies = new Matrix[1 + hidden.length + 1]; // One for input layer, one for each hidden and
																		// one for ouput
		neuronActivationMatricies[0] = new Matrix(inputs, 1);
		for (int i = 0; i < hidden.length; i++) {
			neuronActivationMatricies[i + 1] = new Matrix(hidden[i], 1);
		}
		neuronActivationMatricies[neuronActivationMatricies.length - 1] = new Matrix(output, 1);
	}

	/**
	 * Constructor creates non-mutable recreation of a neural network given a neural
	 * network
	 * 
	 * @param nn A neural network object for recreation
	 */
	public NeuralNetwork(NeuralNetwork nn) {
		LEARNINGRATE = nn.GetLearningRate();

		int[] structure = nn.GetStructure();
		int[] hidden = new int[structure.length - 2];
		for (int i = 0; i < hidden.length; i++) {
			hidden[i] = structure[i + 1];
		}
		this.NumberOfInputNodes = structure[0];
		this.HiddenNodes = hidden;
		this.NumberOfOutputNodes = structure[structure.length - 1];

		weightsMatricies = nn.GetWeights();
		biasMatricies = nn.GetBiases();

		neuronActivationMatricies = new Matrix[1 + hidden.length + 1]; // One for input layer, one for each hidden and
																		// one for ouput
		neuronActivationMatricies[0] = new Matrix(this.NumberOfInputNodes, 1);
		for (int i = 0; i < hidden.length; i++) {
			neuronActivationMatricies[i + 1] = new Matrix(hidden[i], 1);
		}
		neuronActivationMatricies[neuronActivationMatricies.length - 1] = new Matrix(this.NumberOfOutputNodes, 1);
	}

	/**
	 * Constructor creates neural network given the path to a json file previously
	 * generated by this library. JSON interpretor is written from scratch by me.
	 * 
	 * @param directoryPath Path to the directory in which the JSON file is stored
	 * 
	 * @param name          The name of the JSON file used to create the neural
	 *                      network
	 */
	public NeuralNetwork(String directoryPath, String name) {
		int inputs;
		int[] hidden;
		int outputs;
		// -----Imports the Data from File-----
		List<String> lines = new ArrayList<String>();
		String path = directoryPath + "\\" + name + ".json";
		try {
			lines = Files.readAllLines(Paths.get(path), StandardCharsets.UTF_8);
		} catch (IOException e) {
			System.out.println("--Error durring file save--");
		}

		// condense down to one line
		String line = "";
		for (int i = 0; i < lines.size(); i++) {
			line += lines.get(i);
		}

		String curValue = "";

		int structureIndex = line.indexOf("structure") + 13;
		int weightsIndex = line.indexOf("weights") + 11;
		int biasesIndex = line.indexOf("biases") + 10;
		int characterIndex = structureIndex;
		boolean findingStructure = true;
		List<Integer> struct = new ArrayList<Integer>();
		while (findingStructure) {
			char curChar = line.charAt(characterIndex);
			if (curChar == ' ') {
				// it must be a new number
				curValue = "";
			} else if (curChar == ']') {
				// Done the stucture and ended curent value
				try {
					struct.add(Integer.parseInt(curValue));
				} catch (NumberFormatException e) {
					System.out.println("Integer Parse Error");
					e.printStackTrace();
				}
				curValue = "";
				findingStructure = false;
			} else if (curChar == ',') {
				// ended curent value
				try {
					struct.add(Integer.parseInt(curValue));
				} catch (NumberFormatException e) {
					System.out.println("Integer Parse Error");
					e.printStackTrace();
				}
				curValue = "";
			} else {
				// character must be either a number or a dot
				curValue += curChar;
			}
			characterIndex++;
		}

		inputs = struct.get(0);
		outputs = struct.get(struct.size() - 1);
		hidden = new int[struct.size() - 2];
		for (int i = 0; i < hidden.length; i++) {
			hidden[i] = struct.get(i + 1);
		}

		this.NumberOfInputNodes = inputs;
		this.HiddenNodes = hidden;
		this.NumberOfOutputNodes = outputs;

		// moves current index past weight lable ex "0":
		characterIndex = line.substring(weightsIndex).indexOf("\"0") + 4 + weightsIndex;

		weightsMatricies = new Matrix[hidden.length + 1];
		weightsMatricies[0] = new Matrix(hidden[0], inputs, line.substring(characterIndex));
		characterIndex = line.substring(weightsIndex).indexOf("\"1") + 4 + weightsIndex;
		for (int i = 1; i < weightsMatricies.length - 1; i++) {
			weightsMatricies[i] = new Matrix(hidden[i], hidden[i - 1], line.substring(characterIndex));
			characterIndex = line.substring(weightsIndex).indexOf("\"" + (i + 1)) + 4 + weightsIndex;
		}
		weightsMatricies[weightsMatricies.length - 1] = new Matrix(outputs, hidden[hidden.length - 1],
				line.substring(characterIndex));

		characterIndex = line.substring(biasesIndex).indexOf("\"0") + 4 + biasesIndex;
		biasMatricies = new Matrix[hidden.length + 1];
		for (int i = 0; i < hidden.length; i++) {
			biasMatricies[i] = new Matrix(hidden[i], 1, line.substring(characterIndex));
			characterIndex = line.substring(biasesIndex).indexOf("\"" + (i + 1)) + 4 + biasesIndex;
		}
		biasMatricies[biasMatricies.length - 1] = new Matrix(outputs, 1, line.substring(characterIndex));

		neuronActivationMatricies = new Matrix[1 + hidden.length + 1]; // One for input layer, one for each hidden and
																		// one for ouput
		neuronActivationMatricies[0] = new Matrix(inputs, 1);
		for (int i = 0; i < hidden.length; i++) {
			neuronActivationMatricies[i + 1] = new Matrix(hidden[i], 1);
		}
		neuronActivationMatricies[neuronActivationMatricies.length - 1] = new Matrix(outputs, 1);
	}

	/**
	 * Creates a JSON file based on structure and values within a neural network.
	 * Used in conjunction with constructor above.
	 * 
	 * @param directoryPath Path to the directory in which the JSON file will be
	 *                      stored
	 * 
	 * @param name          The name of the JSON file
	 */
	public void Export(String directoryPath, String name) {
		try {
			File save = new File(directoryPath + "\\" + name + ".json");
			if (!save.createNewFile()) {
				System.out.println("File already exist, overwiting file!");
			}
			FileWriter writer = new FileWriter(save);
			writer.write("{\"structure\": [" + NumberOfInputNodes);
			for (int i = 0; i < HiddenNodes.length; i++) {
				writer.write(", " + HiddenNodes[i]);
			}
			writer.write(", " + NumberOfOutputNodes + "],");
			writer.write("\"weights\": {");
			for (int i = 0; i < weightsMatricies.length; i++) {
				writer.write("\"" + i + "\": ");
				writer.write(weightsMatricies[i].ToString());
				if (i != this.weightsMatricies.length - 1) {
					writer.write(",");
				}
			}
			writer.write("},");
			writer.write("\"biases\": {");
			for (int i = 0; i < biasMatricies.length; i++) {
				writer.write("\"" + i + "\": ");
				writer.write(biasMatricies[i].ToString());
				if (i != this.biasMatricies.length - 1) {
					writer.write(",");
				}
			}
			writer.write("}}");
			writer.close();
		} catch (IOException e) {
			System.out.println("--Error durring file save--");
			e.printStackTrace();
		}
	}

	// ------------ GETTERS AND SETTERS ------------------

	/**
	 * Sets the learning rate
	 * 
	 * @param learningRate The learning rate for the neural network
	 */
	public void SetLearningRate(float learningRate) {
		this.LEARNINGRATE = learningRate;
	}

	/**
	 * Gets the learning rate set on the network
	 * 
	 * @return The learning rate
	 */
	public float GetLearningRate() {
		return LEARNINGRATE;
	}

	/**
	 * Gets the structure of the network and provides it as an array
	 * 
	 * @return // {inputs, hiddenlayer1, hiddenLayer2, ... , hiddenLayerN, outputs}
	 */
	public int[] GetStructure() {
		int[] structure = new int[1 + HiddenNodes.length + 1];
		structure[0] = NumberOfInputNodes;
		for (int i = 0; i < HiddenNodes.length; i++) {
			structure[i + 1] = HiddenNodes[i];
		}
		structure[structure.length - 1] = NumberOfOutputNodes;
		return structure;
	}

	/**
	 * Gets the weights matrices
	 * 
	 * @return The weights matrices
	 */
	public Matrix[] GetWeights() {
		Matrix[] weights = new Matrix[weightsMatricies.length];
		for (int i = 0; i < weights.length; i++) {
			weights[i] = new Matrix(weightsMatricies[i]);
		}
		return weights;
	}

	/**
	 * Get the bias matrices
	 * 
	 * @return The bias matrices
	 */
	public Matrix[] GetBiases() {
		Matrix[] biases = new Matrix[biasMatricies.length];
		for (int i = 0; i < biases.length; i++) {
			biases[i] = new Matrix(biasMatricies[i]);
		}
		return biases;
	}

	// ------------- NEURAL NETWORK MATH --------------

	/**
	 * The Feed Forward accepts an input array and uses this to calculate a output
	 * array using the feed forward alogrithm
	 * 
	 * @param inputArray  A single dimensional array of the input float values
	 *                    normalized to be [0,1]
	 * @param printResult The option specifies if the result should print
	 *                    PRETTYPRINT (prints table), UGLYPRINT (comma seperated),
	 *                    DONTPRINT (results not printed)
	 * @return A single dimensional array of the output values
	 */
	public float[] FeedForward(float[] inputArray, PrintResult printResult) {
		Matrix inputValues = Matrix.FromArray(inputArray);

		Matrix[] neuronValues = new Matrix[weightsMatricies.length];

		// calculates the values of the nodes, weighted sum, added bias and normalized
		// sigmoid
		neuronValues[0] = Matrix.MatrixProduct(weightsMatricies[0], inputValues);
		neuronValues[0].add(biasMatricies[0]);
		neuronValues[0].map(ACTIVATIONFUNCTION);
		for (int i = 1; i < neuronValues.length; i++) {
			neuronValues[i] = Matrix.MatrixProduct(weightsMatricies[i], neuronValues[i - 1]);
			neuronValues[i].add(biasMatricies[i]);
			neuronValues[i].map(ACTIVATIONFUNCTION);
		}

		// Placing Values in Neuron Activation Matrices for external speculation
		neuronActivationMatricies[0] = inputValues;
		for (int i = 1; i < neuronValues.length; i++) {
			neuronActivationMatricies[i + 1] = neuronValues[i];
		}

		if (printResult == PrintResult.PRETTYPRINT) {
			String[] labels = new String[Matrix.Transpose(neuronValues[neuronValues.length - 1]).data[0].length];
			for (int i = 0; i < Matrix.Transpose(neuronValues[neuronValues.length - 1]).data[0].length; i++) {
				labels[i] = Integer.toString(i);
			}
			FormatResult("Neural Network Output", labels,
					Matrix.Transpose(neuronValues[neuronValues.length - 1]).data[0], 7);
		} else if (printResult == PrintResult.UGLYPRINT) {
			String strOutput = "";
			for (int i = 0; i < Matrix.Transpose(neuronValues[neuronValues.length - 1]).data[0].length; i++) {
				strOutput += Float.toString(Matrix.Transpose(neuronValues[neuronValues.length - 1]).data[0][i]);
				strOutput += ",";
			}
			System.out.println(strOutput.substring(0, strOutput.length() - 1));
		}

		// returning the calculated outputs
		return neuronValues[neuronValues.length - 1].ToArray();
	}

	/**
	 * Train tweeks the weight values based on provided training data. Data is split
	 * into batches and after all batches are completed the network can be trained
	 * iteratively.
	 * 
	 * @param allInputsArray  Arrays of inputs values corresponding to expected
	 *                        targets. Each array is one data sample.
	 * @param allTargetsArray Arrays of target values corresponding to input values.
	 *                        Each array is one data sample.
	 * @param batchSize       The size of one batch (should be less than the number
	 *                        of data samples)
	 * @param itterations     How many times the network will itterate over the
	 *                        provided data
	 * @param printProgress   Option to specify if current progress is printed
	 *                        PRINTPROGRESS (Prints current progress)
	 *                        DONTPRINTPROGRESS (doesn't print anything)
	 */
	public void Train(float[][] allInputsArray, float[][] allTargetsArray, int batchSize, int itterations,
			PrintProgress printProgress) {
		BATCHSIZE = batchSize;

		// loop whole training procedure how many ever itteration we are doing
		for (int trainingItterations = 0; trainingItterations < itterations; trainingItterations++) {
			// loop through data taking each batch
			for (int n = 0; n < allInputsArray.length - batchSize; n = n + batchSize) {

				// create data arrays for the current batch
				float[][] inputs = new float[batchSize][allInputsArray[n].length];
				float[][] targets = new float[batchSize][allTargetsArray[n].length];

				for (int i = 0; i < batchSize; i++) {
					for (int f = 0; f < allInputsArray[n].length; f++) {
						inputs[i][f] = allInputsArray[n + i][f];
					}

					for (int f = 0; f < targets[i].length; f++) {
						targets[i][f] = allTargetsArray[n + i][f];
					}
				}

				// run gradient descent on batch
				GradientDescent(inputs, targets);

				// print current progress if specified
				if ((printProgress == PrintProgress.PRINTPROGRESS)
						&& ((n % ((int) (allInputsArray.length / 100))) == 0)) {
					System.out.println("Training: "
							+ Integer.toString((int) (((float) n / (float) allInputsArray.length) * 100)) + "%");
				}
			}
			if (printProgress == PrintProgress.PRINTPROGRESS) {
				System.out.println("Completed  iteration " + (trainingItterations + 1) + " out of " + itterations);
			}
		}
	}

	/**
	 * The actual calculation of weight changes based on data sets. Called by the
	 * training function.
	 * 
	 * @param inputsArray  The array of input values, length is one batch
	 * @param targetsArray The array of target values, length is one batch
	 */
	private void GradientDescent(float[][] inputsArray, float[][] targetsArray) {

		int batchSize = inputsArray.length;
		BATCHSIZE = batchSize;

		Matrix[] weightGradientMatrices = new Matrix[weightsMatricies.length];
		for (int i = 0; i < weightGradientMatrices.length; i++) {
			weightGradientMatrices[i] = new Matrix(weightsMatricies[i].row, weightsMatricies[i].col);
		}
		Matrix[] biasGradientMatricies = new Matrix[biasMatricies.length];
		for (int i = 0; i < biasMatricies.length; i++) {
			biasGradientMatricies[i] = new Matrix(biasMatricies[i].row, biasMatricies[i].col);
		}

		for (int batch = 0; batch < batchSize; batch++) {
			// -------------------Feed Forward Algorithm----------------------
			Matrix inputValues = Matrix.FromArray(inputsArray[batch]);
			Matrix targetValues = Matrix.FromArray(targetsArray[batch]);

			Matrix[] neuronValues = new Matrix[weightsMatricies.length];
			Matrix[] neuronValuesSansActivation = new Matrix[weightsMatricies.length];

			// calculates the values of the nodes, weighted sum, added bias and normalized
			// sigmoid
			neuronValues[0] = Matrix.MatrixProduct(weightsMatricies[0], inputValues);
			neuronValues[0].add(biasMatricies[0]);
			neuronValues[0].map(ACTIVATIONFUNCTION);
			neuronValuesSansActivation[0] = Matrix.MatrixProduct(weightsMatricies[0], inputValues);
			neuronValuesSansActivation[0].add(biasMatricies[0]);
			for (int i = 1; i < neuronValues.length; i++) {
				neuronValues[i] = Matrix.MatrixProduct(weightsMatricies[i], neuronValues[i - 1]);
				neuronValues[i].add(biasMatricies[i]);
				neuronValues[i].map(ACTIVATIONFUNCTION);
				neuronValuesSansActivation[i] = Matrix.MatrixProduct(weightsMatricies[i], neuronValues[i - 1]);
				neuronValuesSansActivation[i].add(biasMatricies[i]);
			}

			// -------------------Gradient Calculations-----------------------
			// C/wL = (2(aL-y)(derSig(zL)))0T(aL-1)
			// C/wL-j = (derSig(zL-1)(C/aL-j))0T(aL-j-1)
			// C/aL = sum(wL*derSig(zL)2(aL-yL))
			// C/aL-j = sum(wL-j*derSig(zL)C/aL-j+1)
			// C/bL = derSig(zL)2(aL-yL)
			// C/bL-j = derSig(zL-j)(aL-j-1)

			// Matrix Creation
			Matrix[] curWeightGradientMatrices = new Matrix[weightsMatricies.length];
			for (int i = 0; i < curWeightGradientMatrices.length; i++) {
				curWeightGradientMatrices[i] = new Matrix(weightsMatricies[i].row, weightsMatricies[i].col);
			}
			Matrix[] curBiasGradientMatricies = new Matrix[biasMatricies.length];
			for (int i = 0; i < curBiasGradientMatricies.length; i++) {
				curBiasGradientMatricies[i] = new Matrix(biasMatricies[i].row, biasMatricies[i].col);
			}

			Matrix preCurWeightGradientMatrices = new Matrix(neuronValues[neuronValues.length - 1].row,
					neuronValues[neuronValues.length - 1].col);

			Matrix curNeuronActivationGradient[] = new Matrix[neuronValues.length];
			for (int i = 0; i < curNeuronActivationGradient.length; i++) {
				curNeuronActivationGradient[i] = new Matrix(neuronValues[i].row, neuronValues[i].col);
			}

			// Matrix Mathematics

			// ERROR RELATED CALCULATIONS

			// --Weights--
			// C/wL = (aL-y)
			preCurWeightGradientMatrices = Matrix.Subtract(neuronValues[neuronValues.length - 1], targetValues);
			// C/wL = 2(aL-y)
			preCurWeightGradientMatrices.multiply(2.0f);
			// C/wL = 2(aL-y)(derSig(zL)
			preCurWeightGradientMatrices
					.multiply((Matrix.Map(neuronValuesSansActivation[neuronValuesSansActivation.length - 1],
							ACTIVATIONDERIVATIVEFUNCTION)));
			// C/wL = (2(aL-y)(derSig(zL)))0T(aL-1)
			curWeightGradientMatrices[curWeightGradientMatrices.length - 1] = Matrix.MatrixProduct(
					preCurWeightGradientMatrices, (Matrix.Transpose(neuronValues[neuronValues.length - 2])));

			// --Bias--
			// C/bL = (aL-y)
			curBiasGradientMatricies[curBiasGradientMatricies.length - 1] = Matrix
					.Subtract(neuronValues[neuronValues.length - 1], targetValues);
			// C/bL = 2(aL-y)
			curBiasGradientMatricies[curBiasGradientMatricies.length - 1].multiply(2.0f);
			// C/bL = 2(aL-y)(derSig(zL)
			curBiasGradientMatricies[curBiasGradientMatricies.length - 1]
					.multiply((Matrix.Map(neuronValuesSansActivation[neuronValuesSansActivation.length - 1],
							ACTIVATIONDERIVATIVEFUNCTION)));

			// --Activations--
			// C/aL = sum(wL*derSig(zL)2(aL-yL))
			// Element wise loop
			for (int k = 0; k < curNeuronActivationGradient[curNeuronActivationGradient.length - 1].row; k++) {
				// Summation loop
				for (int i = 0; i < neuronValues[neuronValues.length - 1].row; i++) {
					float sumValue = 0.0f;
					// This is calculated each time wL*derSig(zL)2(aL-yL)
					sumValue = weightsMatricies[weightsMatricies.length - 1].data[i][k];
					sumValue = sumValue
							* ((Matrix.Map(neuronValuesSansActivation[neuronValuesSansActivation.length - 1],
									ACTIVATIONDERIVATIVEFUNCTION)).data[i][0]);
					sumValue = sumValue
							* (Matrix.Subtract(neuronValues[neuronValues.length - 1], targetValues).data[i][0]);
					// Then added as a sum to the final
					curNeuronActivationGradient[curNeuronActivationGradient.length - 1].data[k][0] += sumValue;
				}
			}

			// BACKPROPAGATION

			// --Activations--
			for (int j = 2; j < neuronValues.length + 1; j++) { // j starts at two because the first one is already
																// calculated and -1 is index
				// Element wise loop
				for (int k = 0; k < curNeuronActivationGradient[curNeuronActivationGradient.length - j].row; k++) {
					// Summation loop
					for (int i = 0; i < neuronValues[neuronValues.length - j + 1].row; i++) {
						float sumValue = 0.0f;
						// Value to be added to the slope wL*derSig(zL)2(aL-yL)
						sumValue = weightsMatricies[weightsMatricies.length - j].data[i][k];
						sumValue = sumValue
								* ((Matrix.Map(neuronValuesSansActivation[neuronValuesSansActivation.length - j],
										ACTIVATIONDERIVATIVEFUNCTION)).data[k][0]);
						sumValue = sumValue
								* (curNeuronActivationGradient[curNeuronActivationGradient.length - j + 1].data[i][0]);

						curNeuronActivationGradient[curNeuronActivationGradient.length - j].data[k][0] += sumValue;
					}
				}
			}

			// --Weights--
			for (int j = 2; j < curWeightGradientMatrices.length; j++) {
				// C/wL-j = (derSig(zL-1)(C/aL-j))0T(aL-j-1)
				preCurWeightGradientMatrices = curNeuronActivationGradient[curNeuronActivationGradient.length - j];
				preCurWeightGradientMatrices
						.multiply(Matrix.Map(neuronValuesSansActivation[neuronValuesSansActivation.length - j],
								ACTIVATIONDERIVATIVEFUNCTION));
				curWeightGradientMatrices[curWeightGradientMatrices.length - j] = Matrix.MatrixProduct(
						preCurWeightGradientMatrices, Matrix.Transpose(neuronValues[neuronValues.length - j]));
			}
			preCurWeightGradientMatrices = curNeuronActivationGradient[0];
			preCurWeightGradientMatrices.multiply(Matrix.Map(inputValues, ACTIVATIONDERIVATIVEFUNCTION));
			curWeightGradientMatrices[0] = Matrix.MatrixProduct(preCurWeightGradientMatrices,
					Matrix.Transpose(inputValues));

			// --Bias--
			for (int j = 2; j < curBiasGradientMatricies.length; j++) {
				// C/bL-j = derSig(zL-j)(aL-j-1)
				curBiasGradientMatricies[curBiasGradientMatricies.length
						- j] = curNeuronActivationGradient[curNeuronActivationGradient.length - j];
				curBiasGradientMatricies[curBiasGradientMatricies.length - j]
						.multiply(Matrix.Map(neuronValues[neuronValues.length - j], ACTIVATIONDERIVATIVEFUNCTION));
			}
			curBiasGradientMatricies[0] = curNeuronActivationGradient[0];
			curBiasGradientMatricies[0].multiply(Matrix.Map(inputValues, ACTIVATIONDERIVATIVEFUNCTION));

			// -----------------ALL Gradients Have Been Calculated------------------

			// ADDING FOR AVERAGE WITH BATCH
			for (int i = 0; i < weightGradientMatrices.length; i++) {
				weightGradientMatrices[i].add(curWeightGradientMatrices[i]);
			}
			for (int i = 0; i < biasGradientMatricies.length; i++) {
				biasGradientMatricies[i].add(curBiasGradientMatricies[i]);
			}
		}
		// AVERAGING FOR BATCH
		for (int i = 0; i < weightGradientMatrices.length; i++) {
			weightGradientMatrices[i].multiply(1.0f / batchSize);
		}
		for (int i = 0; i < biasGradientMatricies.length; i++) {
			biasGradientMatricies[i].multiply(1.0f / batchSize);
		}

		// APPLY LEARNINGRATE
		for (int i = 0; i < weightGradientMatrices.length; i++) {
			weightGradientMatrices[i].multiply(LEARNINGRATE);
		}
		for (int i = 0; i < biasGradientMatricies.length; i++) {
			biasGradientMatricies[i].multiply(LEARNINGRATE);
		}

		// ------------------Tweaking Weights and Biases Based on Averaged
		// Gradients------------------
		for (int i = 0; i < weightsMatricies.length; i++) {
			weightsMatricies[i].subtract(weightGradientMatrices[i]);
		}
		for (int i = 0; i < biasMatricies.length; i++) {
			biasMatricies[i].subtract(biasGradientMatricies[i]);
		}
	}

	/**
	 * Tests the Neural Networks accuracy by providing test data. Specify options to
	 * taylor output.
	 * 
	 * @param inputsArray   The input values of the test data
	 * @param targetsArray  The target values test data corresponding to the input
	 *                      values
	 * @param printProgress Prints the current progress if specified (PRINTPROGRESS
	 *                      or DONTPRINTPROGRESS)
	 * @param resultType    The type of output given from the Neural Network: CHOICE
	 *                      (checks if the highest output node matches index of
	 *                      highest target data), CALCULATION (general numerical
	 *                      calculation)
	 * @param prettyResult  Specifies how the results of the test are printed:
	 *                      PRETTYPRINT (provides table of test info), UGLYPRINT
	 *                      (provides comma seperated format)
	 */
	public void Test(float[][] inputsArray, float[][] targetsArray, PrintProgress printProgress, ResultType resultType,
			PrintResult prettyResult) {
		float averageSureness = 0.0f;
		float correctness = 0.0f;
		int correctOnes = 0;
		float cost = 0.0f;
		float[] averageError = new float[targetsArray[0].length];

		// for each of the data samples add to the running average
		for (int n = 0; n < inputsArray.length; n++) {
			float[] outputs = new float[NumberOfOutputNodes];

			// find what the neural network thinks
			outputs = FeedForward(inputsArray[n], PrintResult.DONTPRINT);
			int strongestChoice = -1;
			int targetChoice = -1;
			float sureness = 0.00f;
			float targetSureness = 0.00f;
			// find error for each result and determine if choices are valid
			for (int f = 0; f < outputs.length; f++) {
				if (outputs[f] > sureness) {
					strongestChoice = f;
					sureness = outputs[f];
				}
				if (targetsArray[n][f] > targetSureness) {
					targetChoice = f;
					targetSureness = targetsArray[n][f];
				}
				cost += Math.pow((outputs[f] - (float) targetsArray[n][f]), 2);
				averageError[f] += (float) Math.pow((outputs[f] - (float) targetsArray[n][f]), 2);
			}
			// increment correctness for result type choice
			if (strongestChoice == targetChoice) {
				correctness = correctness + 1.00f;
				averageSureness += sureness;
				correctOnes++;
			}

			// print current progress if specified to do so
			if (printProgress == PrintProgress.PRINTPROGRESS) {
				if ((n % ((int) (inputsArray.length / 100))) == 0) {
					System.out.println("Testing: "
							+ Integer.toString((int) (((float) n / (float) inputsArray.length) * 100)) + "%");
				}
			}
		}

		// take divide sums by totals to create averages
		correctness = correctness / inputsArray.length;
		averageSureness = averageSureness / correctOnes;
		correctness = correctness * 100.00f;
		averageSureness = averageSureness * 100.00f;

		// print results to console based on result type
		if (resultType == ResultType.CHOICE) {
			if (prettyResult == PrintResult.PRETTYPRINT) {
				FormatResult("TESTING COMPLETE",
						new String[] { "Learning Rate", "Batch Size", "Average Correctness", "Average Sureness",
								"Cost" },
						new float[] { LEARNINGRATE, BATCHSIZE, correctness, averageSureness, cost }, 7);
			} else if (prettyResult == PrintResult.UGLYPRINT) {
				System.out.println(
						BATCHSIZE + "," + LEARNINGRATE + "," + correctness + "," + averageSureness + "," + cost);
			}
		} else {
			if (prettyResult == PrintResult.PRETTYPRINT) {
				FormatResult("TESTING COMPLETE", new String[] { "Learning Rate", "Batch Size", "Cost" },
						new float[] { LEARNINGRATE, BATCHSIZE, cost }, 7);
				String[] labels = new String[targetsArray[0].length];
				for (int i = 0; i < targetsArray[0].length; i++) {
					labels[i] = Integer.toString(i);
				}
				FormatResult("Error By Output", labels, averageError, 5);
			} else if (prettyResult == PrintResult.UGLYPRINT) {
				System.out.println(BATCHSIZE + "," + LEARNINGRATE + "," + cost);
				for (int i = 0; i < averageError.length; i++) {
					averageError[i] = averageError[i] / inputsArray.length;
					System.out.print("," + averageError);
				}
				System.out.println();
			}
		}
	}

	// ----------- DISPLAY ---------
	/**
	 * Prints the values of the weights and biases of the Neural Network to the
	 * console.
	 */
	public void PrintNetwork() {
		for (int i = 0; i < weightsMatricies.length; i++) {
			System.out.println("Layer: " + i);
			System.out.println("Weights: ");
			weightsMatricies[i].show();
			System.out.println("Bias: ");
			biasMatricies[i].show();
		}
	}

	/**
	 * Creates a table formatted nicely and printed to the console
	 * 
	 * @param title  The title of the table
	 * @param labels The headers for each row of the table
	 * @param values The values accociated with the labels
	 * @param places The number of decimal places for the values printed in the
	 *               table
	 */
	private void FormatResult(String title, String[] labels, float[] values, int places) {
		places = Math.max(places, 3);
		if (labels.length == values.length) {
			int longestLabel = -1;
			for (int i = 0; i < labels.length; i++) {
				if (labels[i].length() > longestLabel) {
					longestLabel = labels[i].length();
				}
			}
			title = title.replace(' ', '-');
			int width = Math.max(title.length() + 4, (longestLabel + places + 8));
			String bottomCap = "|";
			String topCap = "|";
			String spacer = "|";
			for (int i = 0; i < width - 2; i++) {
				bottomCap += "-";
				spacer += " ";
			}
			for (int i = 0; i < width - 2; i++) {
				if (i == (width - title.length() - 2) / 2) {
					topCap += title;
					i = i + title.length() - 1;
				} else {
					topCap += "-";
				}
			}
			spacer += "|";
			topCap += "|";
			bottomCap += "|";
			System.out.println(topCap);
			System.out.println(spacer);
			for (int i = 0; i < values.length; i++) {
				String strValue = Float.toString(values[i]);
				String outputString = "-1";
				if (strValue.contains("E")) {
					// Scientific Notation
					int exponent = Integer.parseInt(strValue.substring((strValue.indexOf("E") + 1), strValue.length()));
					if (exponent < -(places - 2)) {
						// Very small number
						outputString = "0.";
						for (int j = 0; j < places - 2; j++) {
							outputString += "0";
						}
					} else if (exponent > (places - 2)) {
						// Very Large Number
						// find the substring starting at the begining and ending either where the E is
						// or places
						// checks if the exponent is greater that 10, 2 places must be reserved
						outputString = strValue.substring(0,
								Math.min(strValue.indexOf("E"), (places - 3 - (((exponent >= 10)) ? 1 : 0))));
						outputString += "E";
						outputString += Integer.toString(exponent);
						if (outputString.length() < (places - 3 - ((exponent >= 10) ? 1 : 0)
								- ((outputString.charAt(0) == '-') ? 1 : 0))) {
							outputString += " ";
						}
					} else if (exponent < 0) {
						// medium small number
						outputString = strValue.substring(0 + ((strValue.charAt(0) == '-') ? 1 : 0),
								1 + ((strValue.charAt(0) == '-') ? 1 : 0));
						outputString = outputString
								+ strValue.substring(2 + ((strValue.charAt(0) == '-') ? 1 : 0), strValue.indexOf("E"));
						for (int j = 0; j < Math.abs(exponent); j++) {
							outputString = "0" + outputString;
						}
						outputString = "0." + outputString;
						if (outputString.charAt(0) == '-') {
							outputString = "-" + outputString;
						}
						outputString = outputString.substring(0, places);
					} else if (exponent > 0) {
						// medium large number
						outputString = strValue.substring(0, 1);
						outputString = outputString + strValue.substring(2, strValue.indexOf("E"));
						for (int j = 0; j < exponent; j++) {
							outputString = outputString + "0";
						}
						outputString = outputString.substring(0, places);
					}
				} else {
					// Not Scientific Notation
					if (strValue.indexOf('.') > (places - 1)) {
						// Larger Number
						int exponent = strValue.indexOf('.') - 1;
						outputString = strValue.substring(0, 1) + "." + strValue.substring(2, strValue.indexOf('.'))
								+ strValue.substring(strValue.indexOf('.') + 1, strValue.length());
						outputString = outputString.substring(0, places - 2);
						outputString = outputString + "E" + exponent;
					} else {
						// Smaller Number
						outputString = strValue;
						for (int j = 0; j < (places - strValue.length()); j++) {
							outputString = outputString + "0";
						}

					}
					// outputString = strValue.substring(0, places);
				}
				String valueLine = "|";
				for (int j = 0; j < (width - 4); j++) {
					int textWidth = places + labels[i].length() + 2;
					if (j == (int) ((width / 2) - (textWidth / 2) - 1)) {
						valueLine += labels[i] + ": " + outputString;
						j = j + labels[i].length() + outputString.length();
					}
					valueLine += " ";
				}
				valueLine += "|";
				System.out.println(valueLine);
				System.out.println(spacer);
			}
			System.out.println(bottomCap);
		} else {
			System.out.println("Error: Format result label array value array mismatched dimensions.");
		}
	}

	// -------- ENUMS ------------

	public enum PrintProgress {
		PRINTPROGRESS,
		DONTPRINTPROGRESS
	}

	public enum PrintResult {
		DONTPRINT,
		PRETTYPRINT,
		UGLYPRINT,
	}

	public enum ResultType {
		CHOICE,
		CALCULATION,
	}

}
