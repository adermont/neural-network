import com.github.adermont.neuralnetwork.math.Value;
import com.github.adermont.neuralnetwork.util.NNUtil;

public class TestValue
{
    public void test1(){
        Value a = new Value(1.0).label("a");
        Value x = new Value(2.0).label("x");
        Value b = new Value(3.0).label("b");
        Value y = a.mul(x).plus(b);
        NNUtil.graphviz(y);
    }

    public void test2(){
        Value x1 = new Value("x1", 2);
        Value x2 = new Value("x2", 0);
        Value w1 = new Value("w1", -3);
        Value w2 = new Value("w2", 1);
        Value b = new Value("b", 6.8813735870195432);
        Value x1w1 = x1.mul(w1, "");
        Value x2w2 = x2.mul(w2, "");
        Value x1w1x2w1 = x1w1.plus(x2w2, "");
        Value n = x1w1x2w1.plus(b, "n");
        Value o = n.tanh().label("o");
        o.backPropagation();
        NNUtil.graphviz(o);
    }

    public static void main(String[] args)
    {
        new TestValue().test2();
    }
}
