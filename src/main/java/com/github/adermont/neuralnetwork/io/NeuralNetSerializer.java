package com.github.adermont.neuralnetwork.io;

import com.github.adermont.neuralnetwork.base.NeuralNet;
import com.github.adermont.neuralnetwork.base.Neuron;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.util.IOUtil;

import java.io.*;
import java.nio.file.Path;

public class NeuralNetSerializer
{
    public void serialize(NeuralNet nn, Path dest) throws IOException
    {
        try (FileOutputStream fos = new FileOutputStream(dest.toFile());
             DataOutputStream out = new DataOutputStream(fos);)
        {
            out.writeInt(nn.inputDim());
            out.writeDouble(nn.learningRate());

            out.writeInt(nn.hiddenLayers().size());
            for (NeuralLayer layer : nn.hiddenLayers())
            {
                writeLayer(out, layer);
            }
            NeuralLayer outputLayer = nn.outputLayer();
            if (outputLayer != null)
            {
                out.writeInt(1);
                writeLayer(out, outputLayer);
            }
            else
            {
                out.writeInt(0);
            }
        }
    }

    public NeuralNet deserialize(Path dest) throws IOException
    {
        NeuralNet nn = null;
        try (FileInputStream fis = new FileInputStream(dest.toFile());
             DataInputStream in = new DataInputStream(fis);)
        {
            int inputDim = in.readInt();
            double learningRate = in.readDouble();

            nn = new NeuralNet(inputDim, learningRate);

            int nbLayers = in.readInt();
            for (int i = 0; i < nbLayers; i++)
            {
                readHiddenLayer(in, nn);
            }

            int isOutLayer = in.readInt();
            if (isOutLayer > 0)
            {
                readOutputLayer(in, nn);
            }
        }
        return nn;
    }

    private static void writeLayer(DataOutputStream out, NeuralLayer layer) throws IOException
    {
        out.writeInt(layer.getNeuronCount());
        out.writeUTF(layer.functions().name());

        writeLayerNeurons(out, layer);
    }

    private static void readHiddenLayer(DataInputStream in, NeuralNet nn) throws IOException
    {
        int nbNeurons = in.readInt();
        String functions = in.readUTF();

        DenseLayer layer = nn.addDenseLayer(nbNeurons, NeuronFunctions.forName(functions));
        readLayerNeurons(in, layer);
    }

    private static void readOutputLayer(DataInputStream in, NeuralNet nn) throws IOException
    {
        int nbNeurons = in.readInt();
        String functions = in.readUTF();

        CollectorLayer collectorLayer = nn.setCollectorLayer(nbNeurons,
                                                             NeuronFunctions.forName(
                                                                     functions));
        readLayerNeurons(in, collectorLayer);
    }

    private static void writeLayerNeurons(DataOutputStream out, NeuralLayer layer)
    throws IOException
    {
        Neuron[] neurons = layer.getNeurons();
        out.writeInt(neurons.length);

        for (Neuron neuron : neurons)
        {
            writeNeuron(out, neuron);
        }
    }

    private static void readLayerNeurons(DataInputStream in, NeuralLayer layer) throws IOException
    {
        Neuron[] neurons = layer.getNeurons();
        int neuronCount = in.readInt();
        for (Neuron neuron : neurons)
        {
            readNeuron(in, neuron);
        }
    }

    private static void writeNeuron(DataOutputStream out, Neuron neuron) throws IOException
    {
        out.writeInt(neuron.getNeuronId());
        IOUtil.writeDoubleArray(out, neuron.getWeightsAsDouble());
        out.writeDouble(neuron.getBias().doubleValue());
    }

    private static void readNeuron(DataInputStream in, Neuron neuron) throws IOException
    {
        neuron.setId(in.readInt());
        double[] weights = IOUtil.readDoubleArray(in);
        neuron.setWeights(weights);
        double bias = in.readDouble();
        neuron.setBias(bias);
    }


}
