package model.based_models.hyperparameter_tuning;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;

import static preprocessing.dataImporter.*;

public class LogisticRegressionTuning {


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

    public void exec() {
        try {
            // Load datasets
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();
            Instances validDataset = validSource.getDataSet();

            // Set class index to the last attribute
            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(validDataset);

            // Hyperparameter tuning
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(new Logistic());
            ps.setNumFolds(10); // 10-fold cross-validation
            ps.addCVParameter("R 1.0E-10 1.0E-8 100"); // Ridge parameter

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