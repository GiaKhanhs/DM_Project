package model.based_models.hyperparameter_tuning;

import model.Command;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class J48Tuning implements Command {
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

    @Override
    public void exec() {
        try {
            // Load datasets
            Instances trainDataset = loadDataset("data\\family\\training_data.arff");
            Instances testDataset = loadDataset("data\\family\\test_data.arff");
            Instances validDataset = loadDataset("data\\family\\validation_data.arff");

            // Set class index to the last attribute
            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(validDataset);

            // Hyperparameter tuning
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(new J48());
            ps.setNumFolds(10); // 10-fold cross-validation

            // Add parameters to be optimized
            ps.addCVParameter("M 2 10 1");

            // Perform cross-validation to find the best parameters on the validation dataset
            ps.buildClassifier(validDataset);

            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", ps.getBestClassifierOptions()));

            // Train the Logistic Regression classifier with the best parameters
            J48 j48 = new J48();
            j48.setOptions(ps.getBestClassifierOptions());
            j48.buildClassifier(trainDataset);

            // Evaluate the classifier on the test dataset
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(j48, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

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