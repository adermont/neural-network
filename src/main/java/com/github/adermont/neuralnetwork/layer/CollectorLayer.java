package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;
import org.apache.commons.lang3.stream.Streams;

import java.util.Arrays;
import java.util.OptionalDouble;

public class CollectorLayer extends DenseLayer
{
    private Value[] output;

    public CollectorLayer(int nbNeurons, NeuronFunctions pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    protected Neuron createNeuron(int id, NeuronFunctions pFunction)
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

    public int getClassification(double[] pP)
    {
        double max = Arrays.stream(pP).max().orElse(0.0);
        for (int i = 0; i < pP.length; i++)
        {
            if (pP[i] == max) {
                return i;
            }
        }
        return -1;
    }
}
