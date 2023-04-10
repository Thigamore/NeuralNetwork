import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.*;

import javax.swing.Action;

import java.util.Random;

public class NeuralNetwork {

    // neural network variables
    private double learningRate;
    private CostFunction cost;

    // layer variables
    private ArrayList<double[]> layers;
    private ArrayList<ActivationFn> activationFns;
    private ArrayList<double[][]> weights;
    private ArrayList<double[]> biases;

    // * ----------------- Constructors
    public NeuralNetwork() {
        this(0, null);
    }

    public NeuralNetwork(double learningRate, CostFunction costFunction) {
        // neural net vars
        this.learningRate = learningRate;
        this.cost = costFunction;

        // layer vars
        layers = new ArrayList<>();
        activationFns = new ArrayList<>();
        weights = new ArrayList<>();
        biases = new ArrayList<>();
    }

    // initialize the neural network once all layers have been added;
    public void init() throws Exception {
        if (layers.size() < 2) {
            throw new Exception("Not enough layers for the neural network.\n2 needed, " + layers.size() + " found.");
        }

        // randomizing weights and biases
        Random rand = new Random();

        for (int i = 0; i < layers.size() - 1; i++) {
            double[][] tempWeights = new double[layers.get(i).length][layers.get(i + 1).length];
            double[] tempBiases = new double[layers.get(i + 1).length];
            for (int j = 0; j < tempWeights.length; j++) {
                for (int k = 0; k < tempWeights[j].length; k++) {
                    // randomize biases and weights between -5 and 5
                    tempWeights[j][k] = (rand.nextDouble() * 10) - 5;
                }
            }
            for (int j = 0; j < tempBiases.length; j++) {
                tempBiases[j] = (rand.nextDouble() * 10) - 5;
            }
            System.out.println(tempBiases.length);
            weights.add(tempWeights);
            biases.add(tempBiases);
        }
    }

    // trains the data using forward/backward propogation
    public void trainData(double[][] inputs, double[][] expecteds, int trainIterations) {
        // go through every input
        for (int i = 0; i < trainIterations; i++) {
            // set input layer to input
            double[] tempLayer = layers.get(0);
            for (int j = 0; j < inputs[i % inputs.length].length; j++) {
                tempLayer[j] = inputs[i % inputs.length][j];
            }
            layers.set(0, tempLayer);

            // forward Propogate
            forwardPropagate();
            // printLayers();
            
            // finds the activation of the output layer
            double[] outputActivation = new double[layers.get(layers.size() - 1).length];
            for(int j = 0; j < layers.get(layers.size() - 1).length; j++) {
                outputActivation[j] = activationFns.get(activationFns.size() - 1).apply(layers.get(layers.size() - 1)[j]);
            }

            // calculate error of forward propogation
            double error = cost.applyGroup(outputActivation, expecteds[i % inputs.length]);
            System.out.println(String.format("Error: %.3f", error));

            // calculate error for each output
            double[] outputError = new double[layers.get(layers.size() - 1).length];
            for(int j = 0; j < outputError.length; j++) {
                outputError[j] = cost.applyFn(expecteds[i % inputs.length][j], outputActivation[j]);
            }
            // backward propogation
            backPropogate(outputError);

            forwardPropagate();
            error = cost.applyGroup(outputActivation, expecteds[i % inputs.length]);
            System.out.println(String.format("Error: %.3f", error));
            System.out.println(Arrays.toString(expecteds[i%inputs.length]));
            System.out.println(Arrays.toString(layers.get(layers.size() - 1)));
        }
    }

    // // returns double with accuracy
    // public double testData() {
        
    // }

    public void forwardPropagate() {
        // go through each layer
        for (int i = 0; i < layers.size() - 1; i++) {
            double[] layerSending = layers.get(i);
            double[] layerReceiving = layers.get(i + 1);
            // go through each node of the recieving layer
            for (int rec = 0; rec < layerReceiving.length; rec++) {
                double sum = 0;
                // go through each weight from the sending to receiving node
                for (int send = 0; send < layerSending.length; send++) {
                    sum += weights.get(i)[send][rec] * activationFns.get(i).apply(layerSending[send]);
                }
                // set sum of corresponding layer using activation fn
                layers.get(i + 1)[rec] = sum + biases.get(i)[rec];
                // layers.get(i + 1)[rec] = activationFns.get(i + 1).apply(sum + biases.get(i)[rec]);
            }
        }
    }

    public void backPropogate(double[] costs) {
        // output layer calc
        ArrayList<double[][]> weightChanges = new ArrayList<>();
        // change in a or (w * zi-1) + b for other layers
        ArrayList<double[]> prevLayerChanges = new ArrayList<>();

        //initialize the weight Changes
        for(int i = 0; i < weights.size(); i++) {
            double [][] tempWeightChanges = new double[weights.get(i).length][weights.get(i)[0].length];
            for(int j = 0; j < weights.get(i).length; j++) {
                for(int k = 0; k < weights.get(i)[j].length; k++) {
                    tempWeightChanges[j][k] = 0;
                }
            }
            weightChanges.add(tempWeightChanges);
        }

        // initialize the prevLayerChanges
        for(int i = 0; i < layers.size(); i++) {
            double[] tempLayerChanges = new double[layers.get(i).length];
            for(int j = 0; j < layers.get(i).length; j++) {
                tempLayerChanges[j] = 0;
            }
            prevLayerChanges.add(tempLayerChanges);
        }

        double[][] outputWeightChange = new double[layers.get(layers.size() - 2).length][layers.get(layers.size() - 1).length];
        // the nodes of the previous layer
        for(int i = 0; i < layers.get(layers.size() - 2).length; i++) {
            double[] prevLayer = layers.get(layers.size() - 2);
            double[] outputLayer = layers.get(layers.size() - 1);
            double sum = 0;
            // nodes of output layer
            for(int j = 0; j < outputLayer.length; j++) {
                // n * zi-1 * derivative of cost * derivative of activation
                outputWeightChange[i][j] = learningRate * prevLayer[i] * costs[j] * activationFns.get(activationFns.size() - 1).applyDerivative(outputLayer[j]);
                // wij * derivative of cost * derivative of activation
                sum += weights.get(weights.size() - 1)[i][j] * costs[j] * activationFns.get(activationFns.size() - 1).applyDerivative(outputLayer[j]);
            }
            prevLayerChanges.get(prevLayerChanges.size() - 1)[i] = sum;
        }
        weightChanges.set(weightChanges.size() - 1, outputWeightChange);

        //! WIP test with only one second layer
        double[][] inputWeightChange = new double[layers.get(0).length][layers.get(1).length];
        for(int i = 0; i < layers.get(0).length; i++) {
            double[] inputLayer = layers.get(0);
            double[] hiddenLayer = layers.get(1);
            for(int j = 0; j < hiddenLayer.length; j++) {
                double a = inputLayer[i];
                inputWeightChange[i][j] = learningRate * inputLayer[i] * prevLayerChanges.get(prevLayerChanges.size() - 1)[j] * activationFns.get(activationFns.size() - 2).applyDerivative(hiddenLayer[j]);
            }
        }
        weightChanges.set(0, inputWeightChange);
        // System.out.println("0: " + Arrays.deepToString(weightChanges.get(0)));
        // System.out.println("1: " + Arrays.deepToString(weightChanges.get(1)))
        for(int i = 0 ; i < weights.size(); i++) {
            for(int j = 0; j < weights.get(i).length; j++) {
                for(int k = 0; k < weights.get(i)[j].length; k++) {
                    weights.get(i)[j][k] += weightChanges.get(i)[j][k];
                }
            }
        }
    }

    // add one layer using created activation fn
    public void addLayer(int numNodes, ActivationFn activationFunction) {
        layers.add(new double[numNodes]);
        activationFns.add(activationFunction);
    }

    // add one layer using precreating activation fns
    public void addLayerStr(int numNodes, String activationFn) {
        switch (activationFn.toLowerCase().trim()) {
            case "sigmoid":
                addLayer(numNodes, ActivationFn.SIGMOID);
                break;
            /*
             * case "relu":
             * addLayer(numNodes, NeuralNetwork::relu);
             * break;
             */
            case "empty":
                addLayer(numNodes, ActivationFn.EMPTY);
                break;
            default:
                System.out.println("Couldn't find function");
                break;
        }
    }

    // add multiple layers
    // ! WIP, make nicer
    public void addLayers(int numNodes, ActivationFn activationFunction, int numLayers) {
        for (int i = 0; i < numLayers; i++) {
            addLayer(numNodes, activationFunction);
        }
    }

    // Setters
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setCost(CostFunction costFunction) {
        this.cost = costFunction;
    }

    // * ------------------ utility functions
    // get the data in the form of [input=0, output=1][input/output round][nodes]
    public static int[][][] getInputData(String path, String split) throws IOException, Exception {
        BufferedReader bf = new BufferedReader(new FileReader(path));
        ArrayList<int[]> inputs = new ArrayList<>();
        ArrayList<int[]> expected = new ArrayList<>();

        // skip initial comments
        String line = bf.readLine();
        while (line.charAt(0) == '/') {
            line = bf.readLine();
        }
        // set length
        int lenIn = line.split(split)[0].length();
        int lenExp = line.split(split)[1].length();

        // get each line
        while (line != null) {
            // skip comments
            if (line.charAt(0) == '/') {
                line = bf.readLine();
                continue;
            }
            // split input
            String[] splitInput = line.split(split);
            int[] tempInput = splitInput[0].chars().toArray();
            int[] tempExpected = splitInput[1].chars().toArray();
            if (tempInput.length != lenIn) {
                throw new Exception("Length of input not consistent.");
            } else if (tempExpected.length != lenExp) {
                throw new Exception("Length of output not consistent.");
            }
            // make input an int
            for (int i = 0; i < tempInput.length; i++) {
                tempInput[i] -= '0';
            }
            for (int i = 0; i < tempExpected.length; i++) {
                tempExpected[i] -= '0';
            }
            inputs.add(tempInput);
            expected.add(tempExpected);
            line = bf.readLine();
        }
        bf.close();
        // return mix of the input and expected
        return new int[][][] { inputs.toArray(new int[inputs.size()][]), expected.toArray(new int[expected.size()][]) };
    }

    private void printWeights() {
        for (int i = 0; i < weights.size(); i++) {
            System.out.println("\nFrom layer " + i + " to layer " + (i + 1));
            for (int j = 0; j < weights.get(i).length; j++) {
                System.out.println("\tNode " + j);
                System.out.print("\t\t");
                for (int k = 0; k < weights.get(i)[j].length; k++) {
                    System.out.print(String.format("%.3f, ", weights.get(i)[j][k]));
                }
                System.out.println();
            }
        }
    }

    private void printLayers() {
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("\nLayer " + i);
            System.out.print("\t");
            for (int j = 0; j < layers.get(i).length; j++) {
                System.out.print(String.format("%.3f, ", layers.get(i)[j]));
            }
        }
        System.out.println();
    }

    public static double[] intArrToDouble(int[] arr) {
        return Arrays.stream(arr).asDoubleStream().toArray();
    }

    public static void main(String[] args) {
        // getting training data
        System.out.println("Getting training data");
        // getting input
        int[][][] inputs = null;
        try {
            inputs = getInputData("training.data", ",");
        } catch (IOException ioe) {
            System.out.println("Couldn't find file.");
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            System.exit(1);
        }
        // conver the int array to double for later
        double[][][] inputDouble = new double[inputs.length][inputs[0].length][inputs[0][0].length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[i].length; j++) {
                for (int k = 0; k < inputs[i][j].length; k++) {
                    inputDouble[i][j][k] = inputs[i][j][k];
                }
            }
        }
        System.out.println("Training data succesfully retrieved");

        // setting up network
        System.out.println("Initializing network");
        double learningRate = 5;
        CostFunction costFn = CostFunction.MSE;
        NeuralNetwork net = new NeuralNetwork(learningRate, costFn);

        // adding layers
        net.addLayerStr(15, "empty");
        net.addLayerStr(2, "sigmoid");
        net.addLayerStr(10, "sigmoid");

        // initialize network (weights and biases)
        try {
            net.init();
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
            System.exit(1);
        }
        System.out.println("Successfully initialized network");

        // Training data
        System.out.println("Training network based on data");
        net.printWeights();
        net.trainData(inputDouble[0], inputDouble[1], 1000);
        System.out.println("Network successfully trained");
    }
}