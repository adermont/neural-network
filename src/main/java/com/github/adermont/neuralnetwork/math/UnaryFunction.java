package com.github.adermont.neuralnetwork.math;

import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

public abstract class UnaryFunction extends Value implements DoubleUnaryOperator
{
    protected Value  operand;
    protected String operator;

    public UnaryFunction(String label, String op, Value operand)
    {
        super(label, 0.0);
        this.operator = op;
        this.operand = operand;
        this.data = applyAsDouble(operand.data.doubleValue());
    }

    public UnaryFunction(String op, Value operand)
    {
        this(op, op, operand);
    }

    public String operator()
    {
        return operator;
    }

    @Override
    public List<Value> children()
    {
        return Arrays.asList(operand);
    }

    protected abstract double derivative();

    @Override
    protected void backward()
    {
        this.operand.addGradient(derivative() * this.grad);
    }

    public void resetGradient()
    {
        super.resetGradient();
        for (Value child : children())
        {
            child.resetGradient();
        }
    }

    @Override
    public boolean equals(Object pO)
    {
        if (this == pO)
        {
            return true;
        }

        if (!(pO instanceof UnaryFunction that))
        {
            return false;
        }

        return new EqualsBuilder().appendSuper(super.equals(pO)).append(operand, that.operand)
                                  .append(operator, that.operator).isEquals();
    }

    @Override
    public int hashCode()
    {
        return new HashCodeBuilder(17, 37).appendSuper(super.hashCode()).append(operand)
                                          .append(operator).toHashCode();
    }
}
