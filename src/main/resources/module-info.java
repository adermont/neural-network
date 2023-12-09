module com.github.adermont.neuralnetwork {
    requires javafx.controls;
    requires javafx.fxml;

    requires org.controlsfx.controls;

    opens com.github.adermont.neuralnetwork to javafx.fxml;
    exports com.github.adermont.neuralnetwork;
}