package com.github.adermont.neuralnetwork.graph;

import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;

import java.util.Arrays;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

public class Graph extends LineChart<Number, Number>
{
    public Graph(){
        this("Graph");
    }

    public Graph(String title)
    {
        super(new NumberAxis(), new NumberAxis());

        getXAxis().setLabel("x");
        getYAxis().setLabel("y");
        setTitle(title);
        setCreateSymbols(false); // Désactive les symboles sur les points
        getXAxis().setAutoRanging(false);
        getYAxis().setAutoRanging(false);
    }

    public void plot(String seriesName, double[] x, double[] y)
    {
        //System.out.printf("%s => %s", Arrays.toString(x), Arrays.toString(y));

        XYChart.Series series = new XYChart.Series();
        series.setName(seriesName);
        for (int i = 0; i < x.length; i++)
        {
            series.getData().add(new XYChart.Data(x[i], y[i]));
        }
        getData().add(series);
    }

    public void plot(String seriesName, double[] x, DoubleUnaryOperator f)
    {
        plot(seriesName, x, Arrays.stream(x).map(f).toArray());
    }

    public void plot(String seriesName, double[] x, double[] y, DoubleBinaryOperator f)
    {

    }

    @Override
    public NumberAxis getXAxis()
    {
        return (NumberAxis) super.getXAxis();
    }

    @Override
    public NumberAxis getYAxis()
    {
        return (NumberAxis) super.getYAxis();
    }
}
