package com.github.adermont.neuralnetwork.math;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

public class UnaryFunction extends Value implements DoubleUnaryOperator
{
    protected String            label;
    protected Value             operand;
    protected DerivableFunction function;

    public UnaryFunction(String label, Value left)
    {
        super(0.0);
        this.label = label;
        this.operand = left;
        this.data = data();
    }

    public UnaryFunction(String label, Value data, DerivableFunction function)
    {
        super(0.0);
        this.label = label;
        this.operand = data;
        this.function = function;
        this.data = applyAsDouble(operand.data.doubleValue());
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return this.function == null ? data.doubleValue() : function.applyAsDouble(operand);
    }

    public String operator()
    {
        return label;
    }

    @Override
    public List<Value> children()
    {
        return Arrays.asList(operand);
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
