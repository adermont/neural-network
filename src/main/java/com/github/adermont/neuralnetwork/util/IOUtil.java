package com.github.adermont.neuralnetwork.util;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class IOUtil
{
    public static void writeDoubleArray(DataOutputStream dos, double[] values) throws IOException
    {
        dos.writeInt(values.length);
        for (double value : values)
        {
            dos.writeDouble(value);
        }
    }

    public static double[] readDoubleArray(DataInputStream dis) throws IOException
    {
        int arraySize = dis.readInt();
        double[] doubleArray = new double[arraySize];
        for (int i = 0; i < arraySize; i++)
        {
            doubleArray[i] = dis.readDouble();
        }
        return doubleArray;
    }
}
