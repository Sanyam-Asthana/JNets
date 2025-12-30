package io.github.jnets.components;

public class Functions {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double ReLU(double x) {
       return x > 0 ? x : 0;
    }

}
