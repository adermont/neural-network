package com.github.adermont.neuralnetwork.base.activation;

import com.github.adermont.neuralnetwork.base.NeuronFunction;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;

public class Tanh implements NeuronFunctions
{
    @Override
    public NeuronFunction neuronActivation()
    {
        return neuron -> neuron.getPreactivation().tanh();
    }

    @Override
    public NeuronFunction derivative()
    {
        return neuron -> new Value(1.0).minus(neuron.getOutput().pow(2));
    }
}