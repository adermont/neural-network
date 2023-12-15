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

    static NeuronFunctions forName(String name){
        if ( IDENTITY.name().equals(name) ) return IDENTITY;
        if ( HEAVISIDE.name().equals(name) ) return HEAVISIDE;
        if ( RELU.name().equals(name) ) return RELU;
        if ( SIGMOID.name().equals(name) ) return SIGMOID;
        if ( TANH.name().equals(name) ) return TANH;
        if ( SOFTMAX.name().equals(name) ) return SOFTMAX;
        return null;
    }

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
