package com.github.adermont.neuralnetwork.math;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.function.UnaryOperator;

public class Exponential extends UnaryFunction implements DoubleUnaryOperator
{
    public Exponential(Value operand)
    {
        super("exp", operand);
        this.operand = operand;
        this.operator = "e^";
        this.data = data();
    }

    @Override
    public double applyAsDouble(double left)
    {
        return StrictMath.exp(left);
    }

    @Override
    protected double derivative()
    {
        // e'(x) = e(x)
        // Here : this.data = exp(x)
        return this.data.doubleValue();
    }

}
