package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleUnaryOperator;

public class Heaviside extends UnaryFunction implements DoubleUnaryOperator
{
    public Heaviside(Value operand)
    {
        super("heaviside", operand);
        this.data = applyAsDouble(operand.doubleValue());
    }

    @Override
    public double applyAsDouble(double operand)
    {
        return operand < 0.0 ? 0.0 : 1.0;
    }

    @Override
    protected double derivative()
    {
        return 0.0;
    }

}
