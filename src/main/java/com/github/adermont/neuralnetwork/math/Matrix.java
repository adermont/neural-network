package com.github.adermont.neuralnetwork.math;

import java.util.Arrays;

public class Matrix
{
    private double[] data;
    private int   rowCount;
    private int   colCount;
    private int   label;

    public Matrix(int rowCount, int colCount)
    {
        this.data = new double[rowCount * colCount];
        this.rowCount = rowCount;
        this.colCount = colCount;
    }

    public double[] flatten()
    {
        return data;
    }

    public int rowCount()
    {
        return this.rowCount;
    }

    public int columnCount()
    {
        return this.colCount;
    }

    public int size()
    {
        return this.rowCount * this.colCount;
    }

    public double getValue(int r, int c)
    {
        return this.data[r * this.colCount + c];
    }

    public void set(int r, int c, double value)
    {
        this.data[r * this.colCount + c] = value;
    }

    public int getLabel()
    {
        return this.label;
    }

    public void setLabel(int label)
    {
        this.label = label;
    }

    public String toString()
    {
        StringBuilder sb = new StringBuilder();
        sb.append("Label: ").append(getLabel()).append(System.lineSeparator());
        sb.append("Data : ").append(System.lineSeparator());
        for (int r = 0; r < rowCount(); r++)
        {
            for (int c = 0; c < columnCount(); c++)
            {
                sb.append(getValue(r, c)).append(" ");
            }
            sb.append(System.lineSeparator());
        }
        return sb.toString();
    }

}