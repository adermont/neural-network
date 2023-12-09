package com.github.adermont.neuralnetwork.math;

import java.util.*;

public class Value
{
    protected Number data;
    protected double grad;

    public Value(Number data)
    {
        this.data = data;
        this.grad = 0;
    }

    public Number data()
    {
        return this.data;
    }

    public double gradient()
    {
        return this.grad;
    }

    public Value[] children(){
        return new Value[0];
    }

    public String operator(){
        return null;
    }

    protected void backward(){
        // does nothing
    }

    public void retroPropagate()
    {
        List<Value> topo = new ArrayList<Value>();
        Set<Value> visited = new HashSet<>();

        buildTopo(this, visited, topo);

        // go one variable at a time and apply the chain rule to get its gradient
        this.grad = 1;

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

    public Value mul(Value other)
    {
        return new Multiplication(this, other);
    }

    public Value mul(Number other)
    {
        return new Multiplication(this, new Value(other));
    }

    public Value plus(Value other)
    {
        return new Addition(this, other);
    }

    public Value plus(Number other)
    {
        return new Addition(this, new Value(other));
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
        return new Power(this, new Value(other));
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
        return new ReLU(this);
    }

    @Override
    public String toString()
    {
        return "Value("+this.data+")";
    }

}
