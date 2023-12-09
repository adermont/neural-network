package com.github.adermont.neuralnetwork.math;

import com.github.adermont.neuralnetwork.base.ActivationFunction;

import java.util.function.DoubleUnaryOperator;

public class UnaryFunction extends Value implements DoubleUnaryOperator
{
    protected String             label;
    protected Value              self;
    protected ActivationFunction function;

    public UnaryFunction(String label, Value left)
    {
        super(0.0);
        this.label = label;
        this.self = left;
        this.data = data();
    }

    public UnaryFunction(String label, Value left, ActivationFunction function)
    {
        super(0.0);
        this.label = label;
        this.self = left;
        this.function = function;
        this.data = data();
    }

    public Number data()
    {
        double v = self.data.doubleValue();
        return function == null ? applyAsDouble(v) : function.applyAsDouble(v);
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return data.doubleValue();
    }

    public String operator()
    {
        return label;
    }

    @Override
    public Value[] children()
    {
        return new Value[]{self};
    }
}
