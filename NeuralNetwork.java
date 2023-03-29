import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.*;
import java.util.Random;

public class NeuralNetwork {

    // neural network variables
    private double learningRate;
    private BiFunction<double[], double[], Double> cost;

    // layer variables
    private ArrayList<double[]> layers;
    private ArrayList<Function<Double, Double>> activationFns;
    private ArrayList<double[][]> weights;
    private ArrayList<double[][]> biases;

    //* ----------------- Constructors
    public NeuralNetwork() {
        this(0, null);
    }

    public NeuralNetwork(double learningRate, BiFunction<double[], double[], Double> costFunction) {
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
            double[][] tempBiases = new double[layers.get(i).length][layers.get(i + 1).length];
            for (int j = 0; j < tempWeights.length; j++) {
                for (int k = 0; k < tempWeights[j].length; k++) {
                    // randomize biases and weights between -5 and 5
                    tempWeights[j][k] = (rand.nextDouble() * 10) - 5;
                    tempBiases[j][k] = (rand.nextDouble() * 10) - 5;
                }
            }
            weights.add(tempWeights);
            biases.add(tempBiases);
        }
    }

    // trains the data using forward/backward propogation
    public void trainData(int[][] inputs, int[][] expecteds) {
        // go through every input
        for(int i = 0; i < inputs.length; i++) {
            // set input layer to input
            double[] tempLayer = layers.get(0);
            for(int j = 0; j < inputs[i].length; j++) {
                tempLayer[j] = inputs[i][j];
            }
            layers.set(0, tempLayer);

            // forward Propogate
            forwardPropagate();
            printLayers();
        }
    }

    public void forwardPropagate() {
        // go through each layer
        for(int i = 0; i < layers.size() - 1; i++) {
            double[] layerSending = layers.get(i);
            double[] layerReceiving = layers.get(i+1);
            // go through each node of the recieving layer
            for(int rec = 0; rec < layerReceiving.length; rec++) {
                double sum = 0;
                // go through each weight from the sending to receiving node
                for(int send = 0; send < layerSending.length; send++) {
                    sum += weights.get(i)[send][rec] * layerSending[send];
                }
                // set sum of corresponding layer using activation fn
                layers.get(i+1)[rec] = activationFns.get(i+1).apply(sum);
            }
        }
    }

    // add one layer using created activation fn
    public void addLayer(int numNodes, Function<Double, Double> activationFunction) {
        layers.add(new double[numNodes]);
        activationFns.add(activationFunction);
    }

    // add one layer using precreating activation fns
    public void addLayerStr(int numNodes, String activationFn) {
        switch (activationFn.toLowerCase().trim()) {
            case "sigmoid":
                addLayer(numNodes, NeuralNetwork::sigmoid);
                break;
            case "relu":
                addLayer(numNodes, NeuralNetwork::relu);
                break;
            default:
                System.out.println("Couldn't find function");
                break;
        }
    }

    // add multiple layers 
    //! WIP, make nicer
    public void addLayers(int numNodes, Function<Double, Double> activationFunction, int numLayers) {
        for(int i = 0 ; i < numLayers; i++) {
            addLayer(numNodes, activationFunction);
        }
    }

    // Setters
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setCost(BiFunction<double[], double[], Double> costFunction) {
        this.cost = costFunction;
    }

    //* ------------------ utility functions
    // get the data in the form of [input=0, output=1][input/output round][nodes]
    public static int[][][] getInputData(String path, String split) throws IOException, Exception {
        BufferedReader bf = new BufferedReader(new FileReader(path));
        ArrayList<int[]> inputs = new ArrayList<>();
        ArrayList<int[]> expected = new ArrayList<>();
        
        // skip initial comments
        String line = bf.readLine();
        while(line.charAt(0) == '/') {
            line = bf.readLine();
        }
        // set length
        int lenIn = line.split(split)[0].length();
        int lenExp = line.split(split)[1].length();

        // get each line
        while(line != null) {
            // skip comments
            if(line.charAt(0) == '/') {
                line = bf.readLine();
                continue;
            }
            // split input
            String[] splitInput = line.split(split);
            int[] tempInput = splitInput[0].chars().toArray();
            int[] tempExpected = splitInput[1].chars().toArray();
            if(tempInput.length != lenIn) {
                throw new Exception("Length of input not consistent.");
            } else if(tempExpected.length != lenExp) {
                throw new Exception("Length of output not consistent.");
            }
            // make input an int
            for(int i = 0; i < tempInput.length; i++) {
                tempInput[i] -= '0';
            }
            for(int i = 0; i < tempExpected.length; i++) {
                tempExpected[i] -= '0';
            }
            inputs.add(tempInput);
            expected.add(tempExpected);
            line = bf.readLine();
        }
        bf.close();
        // return mix of the input and expected
        return new int[][][]{inputs.toArray(new int[inputs.size()][]), expected.toArray(new int[expected.size()][])};
    }

    private void printWeights() {
        for(int i = 0; i < weights.size(); i++) {
            System.out.println("\nFrom layer " + i + " to layer " + (i+1));
            for(int j = 0; j < weights.get(i).length; j++) {
                System.out.println("\tNode " + j);
                System.out.print("\t\t");
                for(int k = 0; k< weights.get(i)[j].length; k++) {
                    System.out.print(String.format("%.3f, ", weights.get(i)[j][k]));
                }
                System.out.println();
            }
        }
    }

    private void printLayers() {
        for(int i = 0; i < layers.size(); i++) {
            System.out.println("\nLayer " + i);
            System.out.print("\t");
            for(int j = 0; j < layers.get(i).length; j++) {
                System.out.print(String.format("%.3f, ", layers.get(i)[j]));
            }
        }
        System.out.println();
    }

    //* ------------------  Default Activation Functions
    static public double sigmoid(double num) {
        return 1/(1+Math.exp(-1 * num));
    }

    static public double relu(double num) {
        return Math.max(0f, num);
    }

    //* ------------------- Default cost functions
    static public double meanSquaredError(double[] actual, double[] expected) {
        double mse = 0;
        for(int i = 0; i < actual.length; i++) {
            mse += (actual[i] - expected[i]) * (actual[i] - expected[i]);
        }
        mse /= actual.length;
        return mse;
    }

    public static void main(String[] args) {
        // getting training data
        System.out.println("Getting training data");
        // getting input
        int[][][] inputs = null;
        try {
            inputs = getInputData("training.data", ",");
        } catch(IOException ioe) {
            System.out.println("Couldn't find file.");
        } catch(Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
            System.exit(1);
        }
        System.out.println("Training data succesfully retrieved");

        // setting up network
        System.out.println("Initializing network");
        double learningRate = 0.1;
        BiFunction<double[], double[], Double> costFn = NeuralNetwork::meanSquaredError;
        NeuralNetwork net = new NeuralNetwork(learningRate, costFn);

        // adding layers
        net.addLayer(15, null);
        net.addLayerStr(2, "sigmoid");
        net.addLayerStr(10, "sigmoid");
        
        // initialize network (weights and biases)
        try {
            net.init();
        } catch(Exception ex) {
            System.out.println(ex.getMessage());
            System.exit(1);
        }
        System.out.println("Successfully initialized network");

        // Training data
        System.out.println("Training network based on data");
        net.printWeights();
        net.trainData(inputs[0], inputs[1]);
        System.out.println("Network successfully trained");
    }
}