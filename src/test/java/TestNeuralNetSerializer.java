import com.github.adermont.neuralnetwork.base.NeuralNet;
import com.github.adermont.neuralnetwork.base.NeuronFunctions;
import com.github.adermont.neuralnetwork.layer.NeuralLayer;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;

public class TestNeuralNetSerializer
{
    @Test
    public void testSerialize() throws IOException
    {
        NeuralNet nn = new NeuralNet(5);
        nn.setCollectorLayer(10, NeuronFunctions.RELU);

        Path file = Files.createTempFile("testSerialize", ".tmp");
        file.toFile().deleteOnExit();
        Assertions.assertDoesNotThrow(() -> nn.save(file));

        NeuralNet load = NeuralNet.load(file);
        Assertions.assertEquals(load, nn);
    }
}
