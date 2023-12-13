package com.github.adermont.neuralnetwork.base.activation;

import com.github.adermont.neuralnetwork.base.NeuralLayerFunction;
import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.base.NeuronFunction;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;

public class SoftMax implements NeuronFunctions
{
    @Override
    public NeuralLayerFunction layerActivation()
    {
        return layer -> {
            Neuron[] neurons = layer.getNeurons();
            Value[] regularizedOutputs = new Value[neurons.length];
            double sumExp = 0.0;
            double max = 0.0;
            for (int i = 0; i < neurons.length; i++)
            {
                max = Math.max(max, neurons[i].getPreactivation().doubleValue());
            }
            for (int i = 0; i < neurons.length; i++)
            {
                Value preact = neurons[i].getPreactivation();
                regularizedOutputs[i] = preact.minus(max);
                sumExp += StrictMath.exp(regularizedOutputs[i].doubleValue());
            }
            Value sum = new Value(sumExp);
            for (int i = 0; i < neurons.length; i++)
            {
                neurons[i].setOutput(regularizedOutputs[i].exp().div(sumExp));
            }
        };
    }

    @Override
    public NeuronFunction neuronActivation()
    {
        throw new UnsupportedOperationException();
    }

    @Override
    public NeuronFunction derivative()
    {
        // softmax . (1 - softmax)
        // softmax == neuron.getOutput()
        return neuron -> neuron.getOutput().mul(new Value(1).minus(neuron.getOutput()));
    }
}