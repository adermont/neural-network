package com.github.adermont.neuralnetwork.math;

import com.github.adermont.neuralnetwork.base.BinaryFunction;

public class Addition extends BinaryFunction
{
    public Addition(Value pLeftOperand, Value pRightOperand)
    {
        super("+", pLeftOperand, pRightOperand);
    }

    @Override
    public double applyAsDouble(double left, double right)
    {
        return left + right;
    }

    @Override
    protected void backward()
    {
        self.grad += this.grad;
        other.grad += this.grad;
    }

}
