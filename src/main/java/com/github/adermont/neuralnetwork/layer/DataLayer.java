package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.ActivationFunction;
import com.github.adermont.neuralnetwork.base.DataNeuron;
import com.github.adermont.neuralnetwork.base.Neuron;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;

public class DataLayer extends NeuralLayer
{

    public DataLayer(double[] pData)
    {
        super(pData.length, ActivationFunction.IDENTITY);
    }

    @Override
    protected Neuron createNeuron(int id, ActivationFunction pFunction)
    {
        return new DataNeuron(id);
    }

}
