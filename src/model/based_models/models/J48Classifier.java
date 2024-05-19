package model.based_models.models;

import model.Command;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class J48Classifier implements Command {
    public static void main(String[] args) {
        Command cmd = new J48Classifier();
        cmd.exec();
    }

    @Override
    public void exec() {
        try {
            // Load dataset
            // Load training dataset
            DataSource trainSource = new DataSource("data\\segment-challenge.arff");
            Instances trainDataset = trainSource.getDataSet();

            // Load testing dataset
            DataSource testSource = new DataSource("data\\segment-test.arff");
            Instances testDataset = testSource.getDataSet();

            // Set class index to the last attribute (assuming the last attribute is the class label)
            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }


            // Create and train the NaiveBayes classifier
            J48 j48 = new J48();
            j48.buildClassifier(trainDataset);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(j48, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Save the model
            SerializationHelper.write("j48.model", j48);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
