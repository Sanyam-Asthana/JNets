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

    public Neuron(int noOfWeights, double bias) {
        noOfWeights++;
        this.noOfWeights = noOfWeights;
        this.weights = new double[this.noOfWeights];
        this.weights[noOfWeights - 1] = bias;
    }

    public Neuron(int noOfWeights) {
        noOfWeights++;
        this.noOfWeights = noOfWeights;
        this.weights = new double[this.noOfWeights];
        this.weights[noOfWeights - 1] = 0;
    }

    public void setWeights(double[] weights) {
        if (weights.length != noOfWeights - 1){
            throw new IllegalArgumentException("Weights array must have same length");
        }

        System.arraycopy(weights, 0, this.weights, 0, noOfWeights - 1);
    }

    public void decreaseWeight(int index, double delta) {
        if (index < 0 || index >= noOfWeights - 1) {
            throw new IllegalArgumentException("Index out of bounds");
        }
        this.weights[index] -= delta;
    }

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

   public double activate(double[] inputs) {
        double z = calculateScalarProduct(inputs);
        return Functions.ReLU(z);
   }

    public double activate(double[] inputs, DoubleUnaryOperator activation) {
        double z = calculateScalarProduct(inputs);
        return activation.applyAsDouble(z);
    }
}
