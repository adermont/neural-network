package com.github.adermont.neuralnetwork.layer;

import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.math.Value;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.apache.commons.lang3.stream.Streams;

import java.util.Arrays;

public class CollectorLayer extends DenseLayer
{
    private Value[] output;

    public CollectorLayer(int nbNeurons, NeuronFunctions pFunction)
    {
        super(nbNeurons, pFunction);
    }

    @Override
    protected Neuron createNeuron(int id)
    {
        Neuron neuron = super.createNeuron(id);
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

    public int getOutputClass()
    {
        double[] array = Arrays.stream(output).mapToDouble(Value::doubleValue).toArray();
        return getClass(array);
    }

    public int getClass(double[] array)
    {
        double max = Arrays.stream(array).parallel().max().orElse(0.0);
        for (int i = 0; i < array.length; i++)
        {
            if (array[i] == max) {
                return i;
            }
        }
        return -1;
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

    @Override
    public boolean equals(Object pO)
    {
        if (this == pO)
        {
            return true;
        }

        if (!(pO instanceof CollectorLayer that))
        {
            return false;
        }

        return new EqualsBuilder().appendSuper(super.equals(pO))
                                  .append(getOutput(), that.getOutput()).isEquals();
    }

    @Override
    public int hashCode()
    {
        return new HashCodeBuilder(17, 37).appendSuper(super.hashCode()).append(getOutput())
                                          .toHashCode();
    }
}
