package io.github.jnets.components;
import java.util.function.DoubleUnaryOperator;
import io.github.jnets.components.Functions.*;

public class Neuron {

    public int getNoOfWeights() {
        return noOfWeights - 1;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return this.weights[noOfWeights - 1];
    }

    public void setBias(double bias) {
        this.weights[noOfWeights - 1] = bias;
    }

    private final int noOfWeights;
    private final double[] weights;

    /**
     * Constructor to initialize the Neuron object along with a bias
     * @param noOfWeights The number of weights of the Neuron object
     * @param bias The bias of the Neuron object
     */
    public Neuron(int noOfWeights, double bias) {
        noOfWeights++;
        this.noOfWeights = noOfWeights;
        this.weights = new double[this.noOfWeights];
        this.weights[noOfWeights - 1] = bias;
    }

    /**
     * Constructor to initialize the Neuron object without a bias
     * @param noOfWeights The number of weights of the Neuron object
     */
    public Neuron(int noOfWeights) {
        noOfWeights++;
        this.noOfWeights = noOfWeights;
        this.weights = new double[this.noOfWeights];
        this.weights[noOfWeights - 1] = 0;
    }

    /**
     * Weights initializer for the weights of a Neuron object
     * @param weights Array containing the weight values
     */
    public void setWeights(double[] weights) {
        if (weights.length != noOfWeights - 1){
            throw new IllegalArgumentException("Weights array must have same length");
        }

        System.arraycopy(weights, 0, this.weights, 0, noOfWeights - 1);
    }

    /**
     * Decrement a specific weight by a value
     * @param index The index of the weight according to the weights array
     * @param delta The amount the weight is to be decremented by
     */
    public void decreaseWeight(int index, double delta) {
        if (index < 0 || index >= noOfWeights - 1) {
            throw new IllegalArgumentException("Index out of bounds");
        }
        this.weights[index] -= delta;
    }

    /**
     * Calculates the Scalar Product of the weights vector and the input values vector
     * @param values The values array which are inputs to the Neuron object
     * @return The scalar product result
     */
    public double calculateScalarProduct(double[] values) {
        if (values.length != noOfWeights - 1){
            throw new IllegalArgumentException("Values array must have same length");
        }

        double sum = 0;
        for (int i = 0; i < values.length; i++){
            sum += this.weights[i] * values[i];
        }

        sum += this.weights[noOfWeights - 1];

        return sum;
    }

    /**
     * Calculate the activation output with ReLU as the activation function
     * @param inputs The inputs array which are the inputs to the Neuron object
     * @return The activation output result
     */
    public double activate(double[] inputs) {
        double z = calculateScalarProduct(inputs);
        return Functions.ReLU(z);
    }

    /**
     * Calculate the activation output with a custom activation function
     * @param inputs The inputs array which are the inputs to the Neuron object
     * @param activation The lambda function to be used as the activation function
     * @return The activation output result
     */
    public double activate(double[] inputs, DoubleUnaryOperator activation) {
        double z = calculateScalarProduct(inputs);
        return activation.applyAsDouble(z);
    }
}
