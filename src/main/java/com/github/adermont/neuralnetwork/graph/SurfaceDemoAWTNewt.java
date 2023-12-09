package com.github.adermont.neuralnetwork.graph;

import org.jzy3d.analysis.AWTAbstractAnalysis;
import org.jzy3d.analysis.AnalysisLauncher;
import org.jzy3d.chart.factories.IChartFactory;
import org.jzy3d.chart.factories.NewtChartFactory;
import org.jzy3d.plot3d.primitives.SampleGeom;
import org.jzy3d.plot3d.primitives.Shape;
import org.jzy3d.plot3d.rendering.canvas.Quality;

public class SurfaceDemoAWTNewt extends AWTAbstractAnalysis
{
  public static void main(String[] args) throws Exception {
    SurfaceDemoAWTNewt d = new SurfaceDemoAWTNewt();
    AnalysisLauncher.open(d);
  }


  @Override
  public void init() {
    Shape surface = SampleGeom.surface();

    IChartFactory f = new NewtChartFactory();

    chart = f.newChart(Quality.Advanced());
    chart.getScene().getGraph().add(surface);
  }
}