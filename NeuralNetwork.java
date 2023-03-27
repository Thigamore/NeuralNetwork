import java.util.ArrayList;
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

    // add one layer
    public void addLayer(int numNodes, Function<Double, Double> activationFunction) {
        layers.add(new double[numNodes]);
        activationFns.add(activationFunction);
    }

    // add multiple layers
    //! FIX
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


    // Default Activation Functions
    static public double sigmoid(double num) {
        return 1/(1+Math.exp(-1 * num));
    }

    static public double relu(double num) {
        return Math.max(0f, num);
    }

    public static void main(String[] args) {

    }
}