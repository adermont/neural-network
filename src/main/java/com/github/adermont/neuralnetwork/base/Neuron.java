package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.UnaryFunction;
import com.github.adermont.neuralnetwork.math.Value;

import java.util.function.DoubleUnaryOperator;

public class Neuron
{
    private ActivationFunction function;
    private int                 neuronId;
    private Neuron[]            inputs;
    private double[]            weights;
    private double              bias;
    private Value               output;

    public Neuron(int pNeuronId, ActivationFunction pFunction)
    {
        function = pFunction;
        neuronId = pNeuronId;
    }

    public Neuron(int pNeuronId, ActivationFunction pFunction, double[] pWeights, double pBias)
    {
        this(pNeuronId, pFunction);
        weights = pWeights;
        bias = pBias;
    }

    public Neuron[] getInputs()
    {
        return inputs;
    }

    public void setInputs(Neuron[] pInputs)
    {
        inputs = pInputs;
    }

    public double[] getWeights()
    {
        return weights;
    }

    public void setWeights(double[] pWeights)
    {
        weights = pWeights;
    }

    public double getBias()
    {
        return bias;
    }

    public void setBias(double pBias)
    {
        bias = pBias;
    }

    public int getNeuronId()
    {
        return neuronId;
    }

    public double getOutputAsDouble(){
        return getOutput().data().doubleValue();
    }

    public Value getOutput()
    {
        Value sum = null;
        for (int i = 0; i < inputs.length; i++)
        {
            Value mul = inputs[i].getOutput().mul(weights[i]);
            if (sum != null)
            {
                sum = sum.plus(mul);
            }
            else
            {
                sum = mul;
            }
        }
        sum = sum.plus(bias);

        output = new UnaryFunction("relu", sum, function);

        return output;
    }
}
