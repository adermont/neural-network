package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;
import org.apache.commons.lang3.stream.Streams;

import java.util.Arrays;

public class CollectorLayer extends DenseLayer
{
    private Value[] output;

    public CollectorLayer(int nbNeurons, DerivableFunction pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    protected Neuron createNeuron(int id, DerivableFunction pFunction)
    {
        Neuron neuron = super.createNeuron(id, pFunction);
        return neuron;
    }

    @Override
    public void propagate()
    {
        super.propagate();
        output = Arrays.stream(getNeurons()).map(Neuron::getOutput)
                       .collect(new Streams.ArrayCollector<>(Value.class));
    }

    public Value[] getOutput()
    {
        return output;
    }

    public void resetGradients()
    {
        if (output != null)
        {
            for (Value value : output)
            {
                value.resetGradient();
            }
        }
    }
}
