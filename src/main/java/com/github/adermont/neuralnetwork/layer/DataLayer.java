package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.base.DataNeuron;
import com.github.adermont.neuralnetwork.base.Neuron;

public class DataLayer extends NeuralLayer
{

    public DataLayer(int inputDim)
    {
        super(inputDim, NeuronFunctions.IDENTITY);
    }

    @Override
    protected Neuron createNeuron(int id, NeuronFunctions pFunction)
    {
        return new DataNeuron(this, id);
    }

    @Override
    public void propagate()
    {
        functions.layerActivation().apply(this);
        getNextLayer().ifPresent(layer -> layer.propagate());
    }
}
