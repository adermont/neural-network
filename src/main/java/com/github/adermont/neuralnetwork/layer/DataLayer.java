package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.base.DataNeuron;
import com.github.adermont.neuralnetwork.base.Neuron;

public class DataLayer extends NeuralLayer
{

    public DataLayer(int inputDim, NeuronFunctions functions)
    {
        super(inputDim, functions);
    }

    @Override
    protected Neuron createNeuron(int id)
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
