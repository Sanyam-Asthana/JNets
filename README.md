<div style="text-align: center;">
    <h1>JNets</h1>
A lightweight Java neural network library
    
</div>

# Introduction
**JNets** is a lightweight and high-performant library which helps implement Neural Networks in Java.

# Quick Start
Following is a simple example which initializes a Neuron objects, sets its weights and calculates the dot product and activation output by passing a fixed array of input values.
```java
package io.github.jnets;

import io.github.jnets.components.Neuron;

public class Main {
    static void main() {
            Neuron myNeuron = new Neuron(3);
            double[] weights = new double[] {0.5, 1.0, 1.5};
            myNeuron.setWeights(weights);
            double[] inputs = new double[] {0.1, 0.2, 0.3};
            IO.println("Dot Product: " + myNeuron.calculateScalarProduct(inputs));

            IO.println("Activation value: " + myNeuron.activate(inputs));
    }
}
```

The `activate()` function is overloaded, and an optional lambda function can be passed to calculate the activation output using a custom activation function.
```java
package io.github.jnets;

import io.github.jnets.components.Neuron;

public class Main {
    static void main() {
        Neuron myNeuron = new Neuron(3);
        double[] weights = new double[] {0.5, 1.0, 1.5};
        myNeuron.setWeights(weights);
        double[] inputs = new double[] {0.1, 0.2, 0.3};
        IO.println("Dot Product: " + myNeuron.calculateScalarProduct(inputs));

        IO.println("Activation value: " + myNeuron.activate(inputs, (z) -> z > 0 ? z : 0));
    }
}
```
By default, the `activate()` function has ReLU as the activation function.

An bias can be introduced in the Neuron object by adding an optional parameter to the constructor:
```java
package io.github.jnets;

import io.github.jnets.components.Neuron;

public class Main {
    static void main() {
            Neuron myNeuron = new Neuron(3, 0.3);
            double[] weights = new double[] {0.5, 1.0, 1.5};
            myNeuron.setWeights(weights);
            double[] inputs = new double[] {0.1, 0.2, 0.3};
            IO.println("Dot Product: " + myNeuron.calculateScalarProduct(inputs));

            IO.println("Activation value: " + myNeuron.activate(inputs));
    }
}
```
The default value of the bias (when no bias parameter is passed) is 0.

`decreaseWeight()` can be used to decrease a specific weight of the Neuron object by a specific value:
```java
package io.github.jnets;

import io.github.jnets.components.Neuron;

public class Main {
    static void main() {
        Neuron myNeuron = new Neuron(3, 0.3);
        double[] weights = new double[] {0.5, 1.0, 1.5};
        myNeuron.setWeights(weights);
        double[] inputs = new double[] {0.1, 0.2, 0.3};
        IO.println("Dot Product: " + myNeuron.calculateScalarProduct(inputs));

        IO.println("Activation value: " + myNeuron.activate(inputs));
        myNeuron.decreaseWeight(0, 0.1);
        IO.println("Dot Product: " + myNeuron.calculateScalarProduct(inputs));

        IO.println("Activation value: " + myNeuron.activate(inputs));
    }
}
```

The `Functions.java` class provides some general functions:
- `sigmoid()`
- `ReLU()`

# Architecture Highlights
- **Bias** is implemented by an additional weight which has the same value as the bias. A value of "1" is then multiplied with this additional weight value to result in the final scalar product/activation output.
- Uses `arraycopy()` to copy the weights array passed into the `setWeights()` method to the Neuron object's attribute.

# Future Plans
- Implementing a **Neuron Layer**
- Implementing **Backpropagation**
- Implementing an optimized **Matrix multiplication** using BLAS or the Vector API
- 
