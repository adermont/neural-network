package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.base.DataNeuron;
import com.github.adermont.neuralnetwork.base.Neuron;

public class DataLayer extends NeuralLayer
{

    public DataLayer(double[] pData)
    {
        super(pData.length, DerivableFunction.IDENTITY);
    }

    @Override
    protected Neuron createNeuron(int id, DerivableFunction pFunction)
    {
        return new DataNeuron(id);
    }

}
