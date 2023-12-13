package com.github.adermont.neuralnetwork.base.activation;

import com.github.adermont.neuralnetwork.base.NeuronFunction;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;

public class Relu implements NeuronFunctions
{
    @Override
    public NeuronFunction neuronActivation()
    {
        return neuron -> neuron.getPreactivation().relu();
    }

    @Override
    public NeuronFunction derivative()
    {
        return neuron -> neuron.getOutput().doubleValue() < 0.0 ? new Value(0.0) : new Value(1.0);
    }
}
