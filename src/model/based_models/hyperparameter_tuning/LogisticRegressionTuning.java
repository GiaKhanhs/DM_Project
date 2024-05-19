package model.based_models.hyperparameter_tuning;

import weka.classifiers.functions.Logistic;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class LogisticRegressionTuning {

    public static void main(String[] args) {
        try {
            // Load datasets
            Instances trainDataset = loadDataset("data\\segment-challenge.arff");
            Instances testDataset = loadDataset("data\\segment-test.arff");
            Instances validDataset = loadDataset("data\\segment-challenge.arff");

            // Set class index to the last attribute
            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(validDataset);

            // Hyperparameter tuning
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(new Logistic());
            ps.setNumFolds(5); // 5-fold cross-validation
            ps.addCVParameter("R 1.0E-6 1.0E-3 10"); // Ridge parameter

            // Perform cross-validation to find the best parameters on the validation dataset
            ps.buildClassifier(validDataset);

            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", ps.getBestClassifierOptions()));

            // Train the Logistic Regression classifier with the best parameters
            Logistic lr = new Logistic();
            lr.setOptions(ps.getBestClassifierOptions());
            lr.buildClassifier(trainDataset);

            // Evaluate the classifier on the test dataset
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(lr, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Save the model
            SerializationHelper.write("logistic_regression.model", lr);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Load a dataset from an ARFF file.
     *
     * @param filePath the path to the ARFF file
     * @return the loaded Instances object
     * @throws Exception if there is an error loading the dataset
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

    /**
     * Set the class index to the last attribute if it is not already set.
     *
     * @param dataset the Instances object
     */
    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }
}
