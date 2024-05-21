package preprocessing.helloFX;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.PieChart;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

public class PieChartFactors extends Application {

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        // Create a PieChart
        PieChart pieChart = new PieChart();
        pieChart.getData().add(new PieChart.Data("Work", 478));
        pieChart.getData().add(new PieChart.Data("Education", 393));
        pieChart.getData().add(new PieChart.Data("Self", 321));
        pieChart.getData().add(new PieChart.Data("Family", 170));
        pieChart.getData().add(new PieChart.Data("Love", 138));

        // Enhance the appearance of the chart
        pieChart.setLabelsVisible(true);
        pieChart.getData().forEach(data -> {
            Label label = new Label("");
            data.getNode().setOnMouseEntered(event -> {
                label.setText(String.valueOf((int) data.getPieValue()));
                label.setStyle("-fx-font-size: 30px; -fx-font-weight: bold;");
                label.setLayoutX(event.getSceneX() - label.getWidth() / 2);
                label.setLayoutY(event.getSceneY() - label.getHeight() / 2);
                ((StackPane) primaryStage.getScene().getRoot()).getChildren().add(label);
            });

            data.getNode().setOnMouseExited(event -> {
                ((StackPane) primaryStage.getScene().getRoot()).getChildren().remove(label);
            });
        });

        // Setting the title of the chart
        pieChart.setTitle("Depression Factors");

        // Stack pane to hold the chart
        StackPane root = new StackPane();
        root.getChildren().add(pieChart);

        // Create and set the scene
        Scene scene = new Scene(root, 650, 330);
        primaryStage.setTitle("Depression Factors Chart");
        primaryStage.setScene(scene);
        primaryStage.show();
    }
}

