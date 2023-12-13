package com.github.adermont.neuralnetwork.dataset.mnist;

import com.github.adermont.neuralnetwork.math.Matrix;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;

/**
 * Example :
 * <pre>
 *     MnistDataset mnistDataset = new MnistDataset();
 *     Matrix[] matrices = mnistDataset.loadFromLocal(Path.of("D:", "Datasets", "mnist"));
 * </pre>
 */
public class MnistDataset
{
    public static final int MAGIC_DATA   = 2051;
    public static final int MAGIC_LABELS = 2049;

    public static final String DATA_FILENAME   = "train-images.idx3-ubyte";
    public static final String LABELS_FILENAME = "train-labels.idx1-ubyte";

    private Matrix[] train;

    public Matrix[] loadFromLocal(Path dir) throws IOException
    {
        String absoluteDataPath = dir.resolve(Path.of(DATA_FILENAME)).toFile().getAbsolutePath();
        String absoluteLabelsPath = dir.resolve(Path.of(LABELS_FILENAME)).toFile()
                                       .getAbsolutePath();

        return train = readData(absoluteDataPath, absoluteLabelsPath);
    }

    public Matrix[] loadFromUrl(URL url) throws IOException
    {
        Path destinationPath = Files.createTempFile("downloaded-", ".tmp");
        downloadFile(url, destinationPath);
        Matrix[] result = loadFromLocal(destinationPath);
        Files.delete(destinationPath);
        return result;
    }

    public static void downloadFile(URL fileUrl, Path destination) throws IOException
    {
        System.out.print(
                "Downloading %s to %s...".formatted(fileUrl.toString(), destination.toString()));
        Files.copy(fileUrl.openStream(), destination, StandardCopyOption.REPLACE_EXISTING);
        System.out.println("Successful!".formatted(fileUrl.toString()));
    }

    public Matrix[] get()
    {
        return train;
    }

    public double[][] xtrain(long limit)
    {
        return (double[][]) Arrays.stream(this.train).limit(limit).map(m -> m.flatten())
                                  .toArray(size -> new double[size][]);
    }

    public double[][] ytrain(long limit)
    {
        return (double[][]) Arrays.stream(this.train).limit(limit).map(m -> {
            double[] result = (double[]) new double[10];
            result[m.getLabel()] = 1.0;
            return result;
        }).toArray(size -> new double[size][]);
    }

    public Matrix[] readData(String dataFilePath, String labelFilePath) throws IOException
    {
        try (DataInputStream dataInputStream = new DataInputStream(
                new BufferedInputStream(new FileInputStream(dataFilePath)));
             DataInputStream labelInputStream = new DataInputStream(
                     new BufferedInputStream(new FileInputStream(labelFilePath)));)
        {
            int dataMagicNumber = dataInputStream.readInt();// magic number
            if (dataMagicNumber != MAGIC_DATA)
            {
                throw new IOException(
                        "Magic number mismatch, expected %d, got %d".formatted(MAGIC_DATA,
                                                                               dataMagicNumber));
            }
            int inputSize = dataInputStream.readInt();
            int nRows = dataInputStream.readInt();
            int nCols = dataInputStream.readInt();

            // magic number
            int labelMagicNumber = labelInputStream.readInt();
            if (labelMagicNumber != MAGIC_LABELS)
            {
                throw new IOException(
                        "Magic number mismatch, expected %d, got %d".formatted(MAGIC_LABELS,
                                                                               labelMagicNumber));
            }

            int outputSize = labelInputStream.readInt();
            if (inputSize != outputSize)
            {
                throw new IOException(
                        "inputSize != outputSize (%d vs %d)".formatted(inputSize, outputSize));
            }

            Matrix[] data = new Matrix[inputSize];

            for (int i = 0; i < inputSize; i++)
            {
                Matrix matrix = new Matrix(nRows, nCols);
                matrix.setLabel(labelInputStream.readUnsignedByte());
                for (int r = 0; r < nRows; r++)
                {
                    for (int c = 0; c < nCols; c++)
                    {
                        matrix.set(r, c, dataInputStream.readUnsignedByte());
                    }
                }
                data[i] = matrix;
            }
            return data;
        }
    }
}
