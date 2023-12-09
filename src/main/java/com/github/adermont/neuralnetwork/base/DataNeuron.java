package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;

public class DataNeuron extends Neuron
{
    private double[] data;

    public DataNeuron(int neuronId)
    {
        super(neuronId, ActivationFunction.IDENTITY);
    }

    public void setData(double[] pData)
    {
        data = pData;
    }

    @Override
    public Value getOutput()
    {
        return new Value(data[getNeuronId()]);
    }

}
