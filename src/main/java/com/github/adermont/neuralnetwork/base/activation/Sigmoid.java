package com.github.adermont.neuralnetwork.base.activation;

import com.github.adermont.neuralnetwork.base.NeuronFunction;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;

public class Sigmoid implements NeuronFunctions
{
    @Override
    public NeuronFunction neuronActivation()
    {
        // 1 / (1 + exp(-x))
        return neuron -> new Value(1.0).div(neuron.getPreactivation().neg().exp().plus(1));
    }

    @Override
    public NeuronFunction derivative()
    {
        return neuron -> {
            Value sigma = neuron.getOutput();
            return sigma.mul(new Value(1).minus(sigma));
        };
    }
}
