package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.math.Value;

public class DataNeuron extends Neuron
{
    private double[] data;

    public DataNeuron(NeuralLayer layer, int neuronId)
    {
        super(layer, neuronId, NeuronFunctions.IDENTITY);
        output = new Value("data" + neuronId, 0);
    }

    public void setData(double[] pData)
    {
        data = pData;
        setPreactivation(new Value(data[getNeuronId()]));
    }

}
