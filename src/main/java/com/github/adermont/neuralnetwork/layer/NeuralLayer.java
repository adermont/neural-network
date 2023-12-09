package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.ActivationFunction;
import com.github.adermont.neuralnetwork.base.Neuron;

import java.util.Arrays;
import java.util.Optional;
import java.util.function.DoubleUnaryOperator;

public abstract class NeuralLayer
{
    private Neuron[] neurons;

    private NeuralLayer previousLayer;
    private NeuralLayer nextLayer;

    public NeuralLayer(int nbNeurons, ActivationFunction pFunction)
    {
        neurons = new Neuron[nbNeurons];
        for (int i = 0; i < neurons.length; i++)
        {
            neurons[i] = createNeuron(i, pFunction);
        }
    }

    protected Neuron createNeuron(int id, ActivationFunction pFunction){
        return new Neuron(id, pFunction);
    }

    public Neuron[] getNeurons()
    {
        return neurons;
    }

    public Optional<NeuralLayer> getPreviousLayer()
    {
        return Optional.of(previousLayer);
    }

    public Optional<NeuralLayer> getNextLayer()
    {
        return Optional.ofNullable(nextLayer);
    }

    public void setNextLayer(NeuralLayer pNextLayer)
    {
        nextLayer = pNextLayer;
    }

    public void connect(NeuralLayer pPreviousLayer)
    {
        this.previousLayer = pPreviousLayer;
        pPreviousLayer.setNextLayer(this);
    }

    public void propagate(){
        getNextLayer().ifPresent(layer -> layer.propagate());
    }

    public void setWeights(double[][] pWeights)
    {
        for (int i = 0; i < neurons.length; i++)
        {
            neurons[i].setWeights(pWeights[i]);
        }
    }

    public void setBias(double[] pBias)
    {
        for (int i = 0; i < neurons.length; i++)
        {
            neurons[i].setBias(pBias[i]);
        }
    }
}
