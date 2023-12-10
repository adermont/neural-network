package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;

public class DataNeuron extends Neuron
{
    private double[] data;

    public DataNeuron(int neuronId)
    {
        super(neuronId, DerivableFunction.IDENTITY);
        output = new Value("data"+neuronId, 0);
    }

    public void setData(double[] pData)
    {
        data = pData;
    }

    public void act()
    {
        output.set(data[getNeuronId()]);
    }

}
