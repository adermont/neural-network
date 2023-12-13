package com.github.adermont.neuralnetwork.base.activation;

import com.github.adermont.neuralnetwork.base.NeuronFunction;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;

public class Heaviside implements NeuronFunctions
{
    @Override
    public NeuronFunction neuronActivation()
    {
        return neuron -> neuron.getPreactivation().heaviside();
    }

    @Override
    public NeuronFunction derivative()
    {
        return neuron -> new Value(0.0);
    }
}