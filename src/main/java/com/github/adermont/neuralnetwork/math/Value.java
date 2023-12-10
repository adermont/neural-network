package com.github.adermont.neuralnetwork.math;

import java.util.*;
import java.util.function.DoubleUnaryOperator;

public class Value extends Number
{
    protected static DoubleUnaryOperator inertialFunction = new Sigmoid();

    protected String label;
    protected Number data;
    protected double grad;
    protected double previousGradient = Double.NaN;

    public Value()
    {
        this(0);
    }

    public Value(Number data)
    {
        this(String.valueOf(data), data);
    }

    public Value(String label, Number data)
    {
        this.label = label;
        this.data = data;
        this.grad = 0;
    }

    public void set(double pValue)
    {
        data = pValue;
    }

    public String label()
    {
        return label;
    }

    public Value label(String label)
    {
        if (label != null)
        {
            this.label = label;
        }
        return this;
    }

    public Number data()
    {
        return this.data;
    }

    public double gradient()
    {
        return this.grad;
    }

    public List<Value> children()
    {
        return Collections.emptyList();
    }

    public String operator()
    {
        return null;
    }

    public void backward()
    {
        // does nothing
    }

    public void retroPropagate()
    {
        resetGradient();
        this.grad = 1;

        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();

        buildTopo(this, visited, topo);

        Collections.reverse(topo);
        for (Value v : topo)
        {
            v.backward();
        }
    }

    private void buildTopo(Value v, Set<Value> visited, List<Value> topo)
    {
        if (!visited.contains(v))
        {
            visited.add(v);
            for (Value child : v.children())
            {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }

    public Value mul(Value other, String label)
    {
        return new Multiplication(this, other).label(label);
    }

    public Value mul(Value other)
    {
        return mul(other, null);
    }

    public Value mul(Number other)
    {
        return mul(new Value(other), null);
    }

    public Value plus(Value other, String label)
    {
        return new Addition(this, other).label(label);
    }

    public Value plus(Value other)
    {
        return plus(other, null);
    }

    public Value plus(Number other)
    {
        return plus(new Value(other), null);
    }

    public Value neg()
    {
        return mul(-1);
    }

    public Value pow(Value other)
    {
        return new Power(this, other);
    }

    public Value pow(Number other)
    {
        return new Power(this, new Value("", other));
    }

    public Value minus(Value other)
    {
        return new Addition(this, other.neg());
    }

    public Value minus(Number other)
    {
        return new Addition(this, new Value(other).neg());
    }

    public Value div(Value other)
    {
        return new Multiplication(this, other.pow(-1));
    }

    public Value div(Number other)
    {
        return new Multiplication(this, new Value(other).pow(-1));
    }

    public Value relu()
    {
        return relu(this);
    }

    public Value sigmoid()
    {
        return sigmoid(this);
    }

    public Value linear()
    {
        return linear(this);
    }

    public Value heaviside()
    {
        return heaviside(this);
    }

    public Value tanh()
    {
        return tanh(this);
    }

    public static Value relu(Value v)
    {
        return new ActivationFunction(v, new Relu());
    }

    public static Value sigmoid(Value v)
    {
        return new ActivationFunction(v, new Sigmoid());
    }

    public static Value linear(Value v)
    {
        return new ActivationFunction(v, new Identity());
    }

    public static Value heaviside(Value v)
    {
        return new ActivationFunction(v, new Heaviside());
    }

    public static Value tanh(Value v)
    {
        return new ActivationFunction(v, new Tanh());
    }

    public void addGradient(double pValue)
    {
        this.grad += pValue;
    }

    public void resetGradient()
    {
        this.grad = 0.0;
    }

    public void setGradient(int grad)
    {
        this.grad = grad;
    }

    @Override
    public String toString()
    {
        return label + "(" + this.data + ")";
    }

    @Override
    public int intValue()
    {
        return (int) data;
    }

    @Override
    public long longValue()
    {
        return (long) data;
    }

    @Override
    public float floatValue()
    {
        return (float) data;
    }

    @Override
    public double doubleValue()
    {
        return data.doubleValue();
    }

    public void update(double step)
    {
        if (this.grad != 0.0 && children().isEmpty())
        {
            this.data = this.data.doubleValue() - step * grad;
        }
        previousGradient = grad;
    }

    public double previousGradient(){
        return this.previousGradient;
    }
}
