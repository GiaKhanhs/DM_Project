package model.ensemble_models.models;

import model.Command;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestClassifier implements Command {
    public static void main(String[] args) {
        Command cmd = new RandomForestClassifier();
        cmd.exec();
    }

    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }

    @Override
    public void exec() {
        try {
            // Load datasets
            DataSource trainSource = new DataSource("data\\family\\training_data.arff");
            Instances trainingDataSet = trainSource.getDataSet();

            // Load testing dataset
            DataSource testSource = new DataSource("data\\family\\test_data.arff");
            Instances testingDataSet = testSource.getDataSet();

            // Set class index to the last attribute
            setClassIndex(trainingDataSet);
            setClassIndex(testingDataSet);

            RandomForest forest = new RandomForest();
            forest.buildClassifier(trainingDataSet);

            Evaluation eval = new Evaluation(trainingDataSet);
            eval.evaluateModel(forest, testingDataSet);

            // Print the parameters of the RandomForest model
            System.out.println("RandomForest parameters: " + String.join(" ", forest.getOptions()));

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nPre-tuning RandomForest\n======\n", false));

            // Print the confusion matrix
            System.out.println(eval.toMatrixString("=== Confusion matrix ==="));

            // Print additional evaluation metrics
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("MAE = " + eval.meanAbsoluteError());
            System.out.println("RMSE = " + eval.rootMeanSquaredError());
            System.out.println("RAE = " + eval.relativeAbsoluteError());
            System.out.println("RRSE = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
