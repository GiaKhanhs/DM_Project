package model.ensemble_models.hyperparameter_tuning;

import model.Command;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AdaBoostM1Tuning implements Command {
        public static void main(String[] args) {
        Command cmd = new RandomForestTuning();
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

            // Load validation dataset
            DataSource validSource = new DataSource("data\\family\\validation_data.arff");
            Instances validDataset = validSource.getDataSet();

            // Set class index to the last attribute
            setClassIndex(trainingDataSet);
            setClassIndex(testingDataSet);
            setClassIndex(validDataset);

            double bestAccuracy = -1;
            String[] bestOptions = null;

            // Create an AdaBoostM1 classifier
            AdaBoostM1 ada = new AdaBoostM1();

            // Set up CVParameterSelection for parameter tuning
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(ada);
            ps.setNumFolds(10); // 5-fold cross-validation
            ps.addCVParameter("P 10 100 10"); // Number of iterations (10 to 100 in steps of 10)

            // Perform cross-validation to find the best parameters on the validation dataset
            ps.buildClassifier(validDataset);

            // Create and configure an AdaBoostM1 classifier with the best options
            AdaBoostM1 tempAda = new AdaBoostM1();
            tempAda.setOptions(ps.getBestClassifierOptions());
            tempAda.buildClassifier(validDataset);

            // Evaluate the model on the validation dataset
            Evaluation eval = new Evaluation(validDataset);
            eval.crossValidateModel(tempAda, validDataset, 5, new java.util.Random(1));

            // Update the best options if current options are better
            if (eval.pctCorrect() > bestAccuracy) {
                bestAccuracy = eval.pctCorrect();
                bestOptions = ps.getBestClassifierOptions();
            }

            // Set up the output
            System.out.println("\nPost-tuning AdaBoostM1\n======\n");

            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", bestOptions));

            // Train the AdaBoostM1 classifier with the best parameters
            AdaBoostM1 finalAda = new AdaBoostM1();
            finalAda.setOptions(bestOptions);
            finalAda.buildClassifier(trainingDataSet);

            // Evaluate the classifier on the test dataset
            Evaluation testEval = new Evaluation(trainingDataSet);
            testEval.evaluateModel(finalAda, testingDataSet);

            // Output the evaluation results
            System.out.println(testEval.toSummaryString());

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
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
