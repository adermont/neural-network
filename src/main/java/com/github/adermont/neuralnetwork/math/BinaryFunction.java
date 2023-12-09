package com.github.adermont.neuralnetwork.math;

import java.util.function.DoubleBinaryOperator;

public abstract class BinaryFunction extends Value implements DoubleBinaryOperator
{
    protected String label;
    protected Value self;
    protected Value other;

    public BinaryFunction(String label, Value left, Value right)
    {
        super(0.0);
        this.self = left;
        this.other = right;
        this.label = label;
        this.data = data();
    }

    public String operator(){
        return label;
    }

    @Override
    public Value[] children()
    {
        return new Value[]{self, other};
    }

    public Number data()
    {
        return applyAsDouble(self.data.doubleValue(), other.data.doubleValue());
    }

    @Override
    public String toString()
    {
        return "Value("+this.self+this.label+this.other+")";
    }
}
