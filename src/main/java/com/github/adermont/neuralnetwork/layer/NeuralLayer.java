package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public abstract class NeuralLayer
{
    private Neuron[] neurons;
    private NeuralLayer previousLayer;
    private NeuralLayer nextLayer;

    public NeuralLayer(int nbNeurons, DerivableFunction pFunction)
    {
        neurons = new Neuron[nbNeurons];
        for (int i = 0; i < neurons.length; i++)
        {
            neurons[i] = createNeuron(i, pFunction);
        }
    }

    protected Neuron createNeuron(int id, DerivableFunction pFunction)
    {
        return new Neuron(id, pFunction);
    }

    public Neuron[] getNeurons()
    {
        return neurons;
    }

    public int getNeuronCount()
    {
        return neurons.length;
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

    public void propagate()
    {
        Arrays.stream(getNeurons()).forEach(n -> n.act());
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

    public List<Value> parameters()
    {
        List<Value> result = new ArrayList<>();
        for (int i = 0; i < neurons.length; i++)
        {
            result.addAll(neurons[i].parameters());
        }
        return result;
    }
}
