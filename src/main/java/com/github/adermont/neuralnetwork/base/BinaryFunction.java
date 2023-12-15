package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.math.Value;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleBinaryOperator;

public abstract class BinaryFunction extends Value implements DoubleBinaryOperator
{
    protected Value  self;
    protected Value  other;
    protected String operator;

    public BinaryFunction(String label, String op, Value left, Value right)
    {
        super(label, 0.0);
        this.operator = op;
        this.self = left;
        this.other = right;
        this.data = applyAsDouble(left.doubleValue(), right.doubleValue());
    }

    public BinaryFunction(String op, Value left, Value right)
    {
        this("", op, left, right);
    }

    public String operator()
    {
        return operator;
    }

    @Override
    public List<Value> children()
    {
        return Arrays.asList(self, other);
    }

    public void resetGradient()
    {
        super.resetGradient();
        for (Value child : children())
        {
            child.resetGradient();
        }
    }
}
