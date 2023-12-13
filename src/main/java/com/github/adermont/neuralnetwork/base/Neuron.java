package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.math.Value;

import java.util.*;
import java.util.stream.Collectors;

public class Neuron
{
    private   NeuralLayer     layer;
    private   NeuronFunctions function;
    private   int             neuronId;
    private   Neuron[]        inputs;
    private   Value[]         weights;
    private   Value           bias;
    protected Value           output;
    protected Value           preactivation;

    public Neuron(NeuralLayer layer, int pNeuronId, NeuronFunctions pFunction)
    {
        this.layer = layer;
        this.function = pFunction;
        this.neuronId = pNeuronId;
        this.bias = new Value("n" + neuronId + ".b", 0);
    }

    public Neuron(NeuralLayer layer, int pNeuronId, NeuronFunctions pFunction, double[] pWeights,
                  double pBias)
    {
        this(layer, pNeuronId, pFunction);
        setWeights(pWeights);
        setBias(pBias);
    }

    public Neuron(NeuralLayer layer, int pNeuronId, NeuronFunctions pFunction, double[] pWeights)
    {
        this(layer, pNeuronId, pFunction, pWeights, new Random().nextDouble(-1, 1));
    }

    public Value getPreactivation()
    {
        return preactivation;
    }

    public void setPreactivation(Value p)
    {
        preactivation = p;
    }

    public int getNeuronId()
    {
        return this.neuronId;
    }

    public Neuron[] getInputs()
    {
        return this.inputs;
    }

    public void setInputs(Neuron[] pInputs)
    {
        this.inputs = pInputs;
    }

    public Value[] getWeights()
    {
        return this.weights;
    }

    public double[] getWeightsAsDouble()
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
        List<Value> values = Arrays.stream(weights).collect(Collectors.toList());
        values.add(bias);
        return values;
    }

    public NeuralLayer layer()
    {
        return this.layer;
    }

    public Value getBias()
    {
        return bias;
    }

    public void setBias(double pBias)
    {
        this.bias.set(pBias);
        this.bias.resetGradient();
    }

    public double getOutputAsDouble()
    {
        return getOutput().data().doubleValue();
    }

    public Value getOutput()
    {
        return this.output;
    }

    public void setOutput(Value pOutput)
    {
        this.output = pOutput;
    }
}
