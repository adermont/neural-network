package com.github.adermont.neuralnetwork.math;

import com.github.adermont.neuralnetwork.base.BinaryFunction;

public class Power extends BinaryFunction
{
    public Power(Value value, Value exponent)
    {
        super("^%d".formatted((int) exponent.data.intValue()), value, exponent);
    }

    @Override
    public double applyAsDouble(double left, double right)
    {
        return Math.pow(left, right);
    }

    @Override
    protected void backward()
    {
        self.grad += derivative().doubleValue() * this.grad;
    }

    public Number derivative()
    {
        return (other.data.doubleValue() * StrictMath.pow(self.data.doubleValue(),
                                                          other.data.doubleValue() - 1));
    }

}
