package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.base.activation.*;
import com.github.adermont.neuralnetwork.math.Value;

public interface NeuronFunctions
{
    NeuronFunctions IDENTITY = new Identity();

    NeuronFunctions HEAVISIDE = new Heaviside();

    NeuronFunctions RELU = new Relu();

    NeuronFunctions SIGMOID = new Sigmoid();

    NeuronFunctions TANH = new Tanh();

    NeuronFunctions SOFTMAX = new SoftMax();

    default String name()
    {
        return getClass().getSimpleName().toLowerCase();
    }

    default NeuronFunction preactivation()
    {
        return neuron -> {
            Value sum = null;
            Neuron[] inputs = neuron.getInputs();
            Value[] weights = neuron.getWeights();
            for (int i = 0; i < inputs.length; i++)
            {
                Value mul = inputs[i].getOutput().mul(weights[i]);
                if (sum != null)
                {
                    sum = sum.plus(mul);
                }
                else
                {
                    sum = mul;
                }
            }
            return sum.plus(neuron.getBias());
        };
    }

    default NeuralLayerFunction layerActivation()
    {
        return layer -> {
            for (Neuron neuron : layer.getNeurons())
            {
                neuron.setOutput(neuronActivation().apply(neuron));
            }
        };
    }

    NeuronFunction neuronActivation();

    NeuronFunction derivative();

}
