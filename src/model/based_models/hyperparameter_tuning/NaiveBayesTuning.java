package model.based_models.hyperparameter_tuning;

//import model.Command;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesTuning {
    public static void main(String[] args) {
        try {
            // Load training dataset
            DataSource trainSource = new DataSource("data\\segment-challenge.arff");
            Instances trainDataset = trainSource.getDataSet();

            // Load testing dataset
            DataSource testSource = new DataSource("data\\segment-test.arff");
            Instances testDataset = testSource.getDataSet();

            // Load validation dataset
            DataSource validSource = new DataSource("data\\segment-challenge.arff");
            Instances validDataset = validSource.getDataSet();

            // Set class index to the last attribute (assuming the last attribute is the class label)
            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }
            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }
            if (validDataset.classIndex() == -1) {
                validDataset.setClassIndex(validDataset.numAttributes() - 1);
            }

            // Hyperparameter tuning
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(new NaiveBayes());
            ps.addCVParameter("K 0 1 1"); // Optimize the -K option (Use kernel estimator)
            ps.addCVParameter("D 0 1 1"); // Optimize the -D option (Use supervised discretization)

            // Perform cross-validation to find the best parameters on the validation dataset
            ps.buildClassifier(validDataset);

            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", ps.getBestClassifierOptions()));

            // Train the NaiveBayes classifier with the best parameters
            NaiveBayes nb = new NaiveBayes();
            nb.setOptions(ps.getBestClassifierOptions());
            nb.buildClassifier(trainDataset);

            // Evaluate the classifier
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(nb, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Save the model
            SerializationHelper.write("naivebayes_best.model", nb);

        } catch (java.lang.reflect.InaccessibleObjectException e) {
            System.err.println("Reflection access error: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
