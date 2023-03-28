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
    private BiFunction<Double, Double, Double> cost;

    // layer variables
    private ArrayList<double[]> layers;
    private ArrayList<Function<Double, Double>> activationFns;
    private ArrayList<double[][]> weights;
    private ArrayList<double[][]> biases;

    // Constructors
    public NeuralNetwork() {
        this(0, null);
    }

    public NeuralNetwork(double learningRate, BiFunction<Double, Double, Double> costFunction) {
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

    public void trainData(int[][] inputs, int[][] expecteds) {

    }

    // add one layer
    public void addLayer(int numNodes, Function<Double, Double> activationFunction) {
        layers.add(new double[numNodes]);
        activationFns.add(activationFunction);
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

    public void setCost(BiFunction<Double, Double, Double> costFunction) {
        this.cost = costFunction;
    }

    // utility functions
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

    // Default Activation Functions
    static public double sigmoid(double num) {
        return 1/(1+Math.exp(-1 * num));
    }

    static public double relu(double num) {
        return Math.max(0f, num);
    }

    public static void main(String[] args) {
        int[][][] inputs = null;
        try {
            inputs = getInputData("training.data", ",");
        } catch(IOException ioe) {
            System.out.println("Couldn't find file.");
        } catch(Exception ex) {
            System.out.println(ex.getMessage());
            ex.printStackTrace();
        }
        System.out.println(Arrays.deepToString(inputs));
    }
}