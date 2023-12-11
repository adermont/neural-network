package com.github.adermont.neuralnetwork.base;

import com.github.adermont.neuralnetwork.layer.CollectorLayer;
import com.github.adermont.neuralnetwork.layer.DataLayer;
import com.github.adermont.neuralnetwork.layer.DenseLayer;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import com.github.adermont.neuralnetwork.math.DerivableFunction;
import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;

import java.util.*;

/**
 * A standard neural network.
 */
public class NeuralNet
{
    /**
     * Default learning rate used in case of insufficient batches (less than 10) to have an
     * auto-adaptative learning rate.
     */
    public static final double DEFAULT_LEARNING_RATE = 0.01;

    private int                inputDim;
    private Value[]            output;
    private DataLayer          inputLayer;
    private Deque<NeuralLayer> hiddenLayers;
    private CollectorLayer     outputLayer;
    private List<Value>        parameters;
    private double             learningRate;

    private int epochCount; // Number of Epochs
    private int currentEpoch; // Current Epoch number
    private int batchCount; // Number of batches by epoch
    private int currentBatch; // Current batch number

    private List<NeuralNetListener> listeners;

    /**
     * Creates a new NeuralNet.
     *
     * @param inputDim            Dimension of the input (e.g. for an image of 32x32pixels,
     *                            inputDim=32x32).
     * @param defaultLearningRate Default learning rate (used if the number of batches is not
     *                            sufficient to have an adaptative learning rate).
     */
    public NeuralNet(int inputDim, double defaultLearningRate)
    {
        this.inputDim = inputDim;
        this.hiddenLayers = new ArrayDeque<>();
        this.parameters = new ArrayList<>();
        this.inputLayer = new DataLayer(inputDim);
        this.learningRate = defaultLearningRate;
        this.listeners = new ArrayList<>();
    }

    /**
     * Creates a new NeuralNet with a default learning rate of 0.01.
     *
     * @param inputDim Dimension of the input (e.g. for an image of 32x32pixels, inputDim=32x32).
     */
    public NeuralNet(int inputDim)
    {
        this(inputDim, DEFAULT_LEARNING_RATE);
    }

    /**
     * Adds a listener for events during the learning step.
     *
     * @param pNeuralNetListener The listener.
     */
    public void addListener(NeuralNetListener pNeuralNetListener)
    {
        this.listeners.add(pNeuralNetListener);
    }

    /**
     * Initialize the parameters of a layer and connects it to the previous layer of the NN.
     *
     * @param currentLayer The layer to be initialized.
     */
    private void initLayer(NeuralLayer currentLayer)
    {
        // Connection to the previous layer
        NeuralLayer previousLayer = hiddenLayers.peekLast() == null ? inputLayer : hiddenLayers.peekLast();

        // Connect the layer to the previous one
        currentLayer.connect(previousLayer);

        // Initialize layer's parameters
        initializeLayerWeights(previousLayer, currentLayer);
        initializeLayerBias(currentLayer);
    }

    /**
     * Initializes weights of all neurons of a layer.
     *
     * @param previousLayer Previous layer.
     * @param currentLayer  Layer to be initialized.
     */
    private void initializeLayerWeights(NeuralLayer previousLayer, NeuralLayer currentLayer)
    {
        // Initialize weights with random values (array dim = neurons count of previous layer)
        double[][] weights = new double[currentLayer.getNeuronCount()][];
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] = new Random().doubles(previousLayer.getNeuronCount(), -1, 1).toArray();
        }
        currentLayer.setWeights(weights);
    }

    /**
     * Initializes bias of all neurons of a layer.
     *
     * @param currentLayer Layer to be initialized.
     */
    private void initializeLayerBias(NeuralLayer currentLayer)
    {
        // Initialize bias with random values (array dim = neurons count of CURRENT layer)
        double[] bias = new double[currentLayer.getNeuronCount()];
        for (int i = 0; i < bias.length; i++)
        {
            bias[i] = new Random().nextDouble(-1, 1);
        }
        currentLayer.setBias(bias);
    }

    /**
     * Creates a new dense layer and connects it to the previous layer.
     *
     * @param nbNeurons Neurons count of the dense layer to be added.
     * @param pFunction The activation function of the neurons of the dense layer.
     * @return The dense layer with all neurons initialized with random weights and zero bias.
     */
    public DenseLayer addDenseLayer(int nbNeurons, DerivableFunction pFunction)
    {
        DenseLayer dense = new DenseLayer(nbNeurons, pFunction);
        initLayer(dense);
        hiddenLayers.add(dense);
        return dense;
    }

    /**
     * Creates a dense layer as output layer. The neurons count determines the dimension of your
     * output. The DerivableFunction is mandatory, so if you just want to get output of the last
     * layer, use {@link com.github.adermont.neuralnetwork.math.DerivableFunction#IDENTITY} as the
     * activation function for the collector layer.
     *
     * @param nbNeurons Number of neurons.
     * @param pFunction Activation function.
     * @return The created layer, automatically connected to previous layer.
     */
    public CollectorLayer setCollectorLayer(int nbNeurons, DerivableFunction pFunction)
    {
        CollectorLayer collector = new CollectorLayer(nbNeurons, pFunction);
        initLayer(collector);
        return outputLayer = collector;
    }

    /**
     * Only tries to predict a value according to the current network's parameters. This method does
     * not learn (to switch to learning mode, see
     * {@link #fit(double[][], double[][], int, double)}.
     * <p>
     * Note : see also the loss() function if you wan to estimate the loss level of this
     * prediction.
     * </p>
     *
     * @param pX The input you want to predict an output.
     * @return An array of predicted output values.
     */
    public Value[] predict(double[] pX)
    {
        for (Neuron neuron : inputLayer.getNeurons())
        {
            ((DataNeuron) neuron).setData(pX);
        }
        inputLayer.propagate();
        return this.output = outputLayer.getOutput();
    }

    /**
     * Starts deep learning.
     *
     * @param x         All training inputs.
     * @param y         Expected output for those training data.
     * @param nbEpoch   Number of epochs.
     * @param batchSize Percentage (must be in range [0,1]) of inputs to process by batch (inputs of
     *                  each batch will be chosen randomly).
     */
    public void fit(double[][] x, double[][] y, int nbEpoch, double batchSize)
    {
        batchSize = Math.min(1d, Math.max(batchSize, 0d));

        this.epochCount = nbEpoch;
        this.batchCount = (int) Math.ceil((double) x.length / (x.length * batchSize));

        for (this.currentEpoch = 0; this.currentEpoch < epochCount; this.currentEpoch++)
        {
            fitNextEpoch(x, y);
        }
    }

    /**
     * Starts a new epoch.
     *
     * @param x Training inputs.
     * @param y Expected outputs.
     */
    private void fitNextEpoch(double[][] x, double[][] y)
    {
        listeners.forEach(l -> l.epochStarted(this.currentEpoch+1, this.epochCount));

        List<Double> totalLossByEpoch = new ArrayList<>();

        // Randomize input (and output accordingly)
        NNUtil.shuffle(x, y);

        // Number of inputs for each batch :
        int batchSize = x.length / this.batchCount;

        // Start batches
        int batchStartIndex = 0;
        for (this.currentBatch = 0; this.currentBatch < this.batchCount; this.currentBatch++)
        {
            listeners.forEach(
                    l -> l.batchStarted(currentBatch+1, batchCount, batchSize, batchStartIndex));

            // ==========================================
            // Do not forget to reset all the gradients !
            this.outputLayer.resetGradients();
            // ==========================================

            Value loss = null;
            for (int i = batchStartIndex; i < Math.min(batchSize, x.length); i++)
            {
                // Predict output
                Value[] prediction = predict(x[i]);

                // Accumulate absolute loss during the batch
                loss = loss(y[i], prediction).plus(loss);
            }

            // Now compute the Mean Squared Error (MSE)
            int nbInputs = Math.min(batchSize, x.length) - batchStartIndex;
            final Value mse = loss.div(nbInputs);

            // Back propagation and parameters update
            updateParameters(mse);

            totalLossByEpoch.add(mse.doubleValue());
            listeners.forEach(l -> l.batchTerminated(currentBatch+1, mse));
        }

        listeners.forEach(l -> l.epochTerminated(currentEpoch+1, totalLossByEpoch));
    }

    /**
     * Compares the last output with an expected output value.
     *
     * @param expected The expected output.
     */
    public Value loss(double[] expected, Value[] output)
    {
        Value loss = new Value("", 0);
        String sBatch = "batch" + currentBatch;

        for (int i = 0; i < expected.length; i++)
        {
            Value expectedValue = new Value(expected[i]).label(sBatch + ".expect" + i);
            Value diff = expectedValue.minus(output[i]).label(sBatch + ".diff" + i);
            loss = loss.plus(diff.pow(2)).label(sBatch + ".loss" + i);
        }
        return loss;
    }

    /**
     * @return The last predicted output or null if no prediction or no learning has been made.
     */
    public Value[] getOutput()
    {
        return output;
    }

    /**
     * @return All the network parameters.
     */
    public List<Value> parameters()
    {
        if (parameters.isEmpty())
        {
            hiddenLayers.forEach(layer -> parameters.addAll(layer.parameters()));
            parameters.addAll(outputLayer.parameters());
        }
        return parameters;
    }

    /**
     * Starts a standard gradient descent by backpropagation.
     */
    private void updateParameters(Value loss)
    {
        // Optimize the learning rate : higher values in the beginning accelerates gradient descent
        if (batchCount >= 10)
        {
            //        System.out.printf("- Updating weights, learning rate=%f%n", learningRate);
            learningRate = 1.0 - (0.9 * ((double) currentBatch / (double) batchCount));
        }

        // Start back propagation
        loss.retroPropagate();

        // Update parameters
        List<Value> parameters = parameters();
        for (Value parameter : parameters)
        {
            parameter.update(learningRate);
        }
    }

    public void summary()
    {
        System.out.println("Model: sequential");
        System.out.println("_____________________________________________________________________");
        System.out.println(" Layer (type)\t\t\tOutput shape\t\t\tParam #");
        System.out.println("=====================================================================");
        for (NeuralLayer hiddenLayer : hiddenLayers)
        {
            System.out.println(
                    " %s\t\t\t\t%s\t\t\t\t%d".formatted(hiddenLayer.getClass().getSimpleName(),
                                                        "(None, " + hiddenLayer.getNeuronCount() + ")",
                                                        hiddenLayer.parameters().size()));
            System.out.println();
        }
        System.out.println(" Output (Dense)\t\t\t%s\t\t\t\t%d".formatted(
                "(None, " + outputLayer.getNeuronCount() + ")", outputLayer.parameters().size()));
        System.out.println("=====================================================================");
        System.out.println("Total params        : %d".formatted(parameters().size()));
        System.out.println("Trainable params    : %d".formatted(parameters().size()));
        System.out.println("Non-trainable params: %d".formatted(parameters().size()));
        System.out.println();
    }

}
