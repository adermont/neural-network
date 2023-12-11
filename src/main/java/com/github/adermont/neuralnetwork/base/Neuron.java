package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.ActivationFunction;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class Neuron
{
    private   DerivableFunction function;
    private   int               neuronId;
    private   Neuron[]          inputs;
    private   Value[]           weights;
    private   Value             bias;
    protected Value             output;

    public Neuron(int pNeuronId, DerivableFunction pFunction)
    {
        function = pFunction;
        neuronId = pNeuronId;
        bias = new Value("n" + neuronId + ".b", 0);
    }

    public Neuron(int pNeuronId, DerivableFunction pFunction, double[] pWeights, double pBias)
    {
        this(pNeuronId, pFunction);
        setWeights(pWeights);
        setBias(pBias);
    }

    public Neuron(int pNeuronId, DerivableFunction pFunction, double[] pWeights)
    {
        this(pNeuronId, pFunction, pWeights, new Random().nextDouble(-1, 1));
    }

    public Neuron[] getInputs()
    {
        return this.inputs;
    }

    public void setInputs(Neuron[] pInputs)
    {
        this.inputs = pInputs;
    }

    public double[] getWeights()
    {
        return Arrays.stream(this.weights).mapToDouble(Value::doubleValue).toArray();
    }

    public void setWeights(double[] pWeights)
    {
        this.weights = new Value[pWeights.length];
        for (int i = 0; i < pWeights.length; i++)
        {
            this.weights[i] = new Value("n" + this.neuronId + ".w" + i, pWeights[i]);
        }
    }

    public List<Value> parameters()
    {
        List<Value> result = new ArrayList<>();
        for (Value weight : this.weights)
        {
            result.add(weight);
        }
        result.add(this.bias);
        return result;
    }

    public double getBias()
    {
        return bias.doubleValue();
    }

    public void setBias(double pBias)
    {
        this.bias.set(pBias);
        this.bias.resetGradient();
    }

    public int getNeuronId()
    {
        return this.neuronId;
    }

    public void act()
    {
        Value sum = null;
        for (int i = 0; i < this.inputs.length; i++)
        {
            Value mul = this.inputs[i].getOutput().mul(this.weights[i]);
            if (sum != null)
            {
                sum = sum.plus(mul);
            }
            else
            {
                sum = mul;
            }
        }
        sum = sum.plus(this.bias);

        this.output = new ActivationFunction(sum, this.function);
    }

    public double getOutputAsDouble()
    {
        return getOutput().data().doubleValue();
    }

    public Value getOutput()
    {
        return this.output;
    }
}
