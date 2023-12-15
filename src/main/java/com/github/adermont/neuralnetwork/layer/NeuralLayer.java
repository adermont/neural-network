package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

public abstract class NeuralLayer
{
    private Neuron[]        neurons;
    private NeuralLayer     previousLayer;
    private NeuralLayer     nextLayer;
    protected NeuronFunctions functions;
    private double          output;

    public NeuralLayer(int nbNeurons, NeuronFunctions pFunctions)
    {
        neurons = new Neuron[nbNeurons];
        functions = pFunctions;
        for (int i = 0; i < neurons.length; i++)
        {
            neurons[i] = createNeuron(i);
        }
    }

    protected Neuron createNeuron(int id)
    {
        return new Neuron(this, id);
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
        // Preactivation
        for (int i = 0; i < neurons.length; i++)
        {
            Neuron n = neurons[i];
            Value preact = functions.preactivation().apply(n);
            n.setPreactivation(preact);
        }

        // Activation
        functions.layerActivation().apply(this);

        sumOutput();

        getNextLayer().ifPresent(layer -> layer.propagate());
    }

    private void sumOutput()
    {
        //this.output = Arrays.stream(getNeurons()).mapToDouble(Neuron::getOutputAsDouble).sum();
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

    public NeuronFunctions functions()
    {
        return functions;
    }

    @Override
    public boolean equals(Object pO)
    {
        if (this == pO)
        {
            return true;
        }

        if (!(pO instanceof NeuralLayer that))
        {
            return false;
        }

        return new EqualsBuilder().append(getNeurons(), that.getNeurons())
                                  .append(functions, that.functions).isEquals();
    }

    @Override
    public int hashCode()
    {
        return new HashCodeBuilder(17, 37).append(getNeurons()).append(functions).toHashCode();
    }

}
