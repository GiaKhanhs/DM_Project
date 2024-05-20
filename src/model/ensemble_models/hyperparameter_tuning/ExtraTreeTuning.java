package model.ensemble_models.hyperparameter_tuning;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.ExtraTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class ExtraTreeTuning {
    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }
    public static double tuneAndEvaluate(String trainPath, String testPath, String validPath) throws Exception {
        // Load datasets
        DataSource trainSource = new DataSource(trainPath);
        Instances trainingDataSet = trainSource.getDataSet();

        // Load testing dataset
        DataSource testSource = new DataSource(testPath);
        Instances testingDataSet = testSource.getDataSet();

        // Load validation dataset
        DataSource validSource = new DataSource(validPath);
        Instances validDataset = validSource.getDataSet();

        // Set class index to the last attribute
        setClassIndex(trainingDataSet);
        setClassIndex(testingDataSet);
        setClassIndex(validDataset);

        double bestAccuracy = -1;
        String[] bestOptions = null;

        // Create ExtraTree Classifier
        ExtraTree ET = new ExtraTree();

        // Set up CVParameterSelection for parameter tuning
        CVParameterSelection ps = new CVParameterSelection();
        ps.setClassifier(ET);
        ps.setNumFolds(5); // 10-fold cross-validation
        ps.addCVParameter("K 1 10 1"); // Number of attributes to randomly choose at a node
        ps.addCVParameter("N 1 10 1"); // Minimum number of instances required at a node for splitting

        // Perform cross-validation to find the best parameters on the validation dataset
        ps.buildClassifier(validDataset);

        // Create and configure a RandomTree classifier with the best options
        ExtraTree tempET = new ExtraTree();
        tempET.setOptions(ps.getBestClassifierOptions());
        tempET.buildClassifier(validDataset);

        // Evaluate the model on the validation dataset
        Evaluation eval = new Evaluation(validDataset);
        eval.crossValidateModel(tempET, validDataset, 5, new java.util.Random(1));

        // Update the best options if current options are better
        if (eval.pctCorrect() > bestAccuracy) {
            bestAccuracy = eval.pctCorrect();
            bestOptions = ps.getBestClassifierOptions();
        }

        // Print the best parameters
        System.out.println("Best Parameters: " + String.join(" ", bestOptions));

        // Train the RandomTree classifier with the best parameters
        ExtraTree finalET = new ExtraTree();
        finalET.setOptions(bestOptions);
        finalET.buildClassifier(trainingDataSet);

        // Evaluate the classifier on the test dataset
        Evaluation testEval = new Evaluation(trainingDataSet);
        testEval.evaluateModel(finalET, testingDataSet);

        // Output the evaluation results
        System.out.println(testEval.toSummaryString("\nResults\n======\n", false));

        // Print the confusion matrix
        System.out.println(testEval.toMatrixString("=== Confusion matrix ==="));

        // Print additional evaluation metrics
        System.out.println("Correct % = " + testEval.pctCorrect());
        System.out.println("Incorrect % = " + testEval.pctIncorrect());
        System.out.println("AUC = " + testEval.areaUnderROC(1));
        System.out.println("Kappa = " + testEval.kappa());
        System.out.println("MAE = " + testEval.meanAbsoluteError());
        System.out.println("RMSE = " + testEval.rootMeanSquaredError());
        System.out.println("RAE = " + testEval.relativeAbsoluteError());
        System.out.println("RRSE = " + testEval.rootRelativeSquaredError());
        System.out.println("Precision = " + testEval.precision(1));
        System.out.println("Recall = " + testEval.recall(1));
        System.out.println("F-Measure = " + testEval.fMeasure(1));
        System.out.println("Error Rate = " + testEval.errorRate());
        System.out.println(testEval.toClassDetailsString());

        return testEval.pctCorrect();
    }
}
