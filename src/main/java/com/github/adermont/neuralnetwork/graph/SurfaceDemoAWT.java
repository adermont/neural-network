package com.github.adermont.neuralnetwork.graph;

import org.jzy3d.analysis.AWTAbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.IChartFactory;
import org.jzy3d.chart.factories.IFrame;
import org.jzy3d.chart.factories.NewtChartFactory;
import org.jzy3d.colors.Color;
import org.jzy3d.colors.ColorMapper;
import org.jzy3d.colors.colormaps.ColorMapRainbow;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Func3D;
import org.jzy3d.plot3d.builder.SurfaceBuilder;
import org.jzy3d.plot3d.builder.concrete.OrthonormalGrid;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

import java.io.File;

public class SurfaceDemoAWT extends AWTAbstractAnalysis
{
    public static void main(String[] args) throws Exception
    {
        SurfaceDemoAWT d = new SurfaceDemoAWT();
        AnalysisLauncher.open(d);
    }

    @Override
    public void init()
    {
        // Define a function to plot
        Func3D func = new Func3D((x, y) -> x * Math.sin(x * y));
        Range range = new Range(-3, 3);
        int steps = 80;

        // Create the object to represent the function over the given range.
        final Shape surface = new SurfaceBuilder().orthonormal(new OrthonormalGrid(range, steps),
                                                               func);
        surface.setColorMapper(
                new ColorMapper(new ColorMapRainbow(), surface, new Color(1, 1, 1, .5f)));
        surface.setFaceDisplayed(true);
        surface.setWireframeDisplayed(true);
        surface.setWireframeColor(Color.BLACK);

        // Create a chart
        IChartFactory f = new NewtChartFactory();
        this.chart = f.newChart(Quality.Advanced().setHiDPIEnabled(true));
        chart.getScene().getGraph().add(surface);
        chart.getScreenshotKey().setFilename(new File("screenshot.png").getAbsolutePath());
        IFrame frame = chart.open();
        chart.addMouse();

    }
}