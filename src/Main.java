//import model.ensemble_models.RandomForestClassifier;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
    public static void main(String[] args) {
        try {
            DataSource trainSource = new DataSource("data\\segment-challenge.arff");
            Instances trainingDataSet = trainSource.getDataSet();

            // Load testing dataset
            DataSource testSource = new DataSource("data\\segment-test.arff");
            Instances testingDataSet = testSource.getDataSet();

            // Set class index to the last attribute (assuming the last attribute is the class label)
            if (trainingDataSet.classIndex() == -1) {
                trainingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);
            }

            if (testingDataSet.classIndex() == -1) {
                testingDataSet.setClassIndex(testingDataSet.numAttributes() - 1);
            }


            RandomForest forest = new RandomForest();
            AdaBoostM1 ada = new AdaBoostM1();
            NaiveBayes naive = new NaiveBayes();

            /** */
            //forest.buildClassifier(trainingDataSet);
            ada.buildClassifier(trainingDataSet);

            /**
             * train the alogorithm with the training data and evaluate the
             * algorithm with testing data
             */
            Evaluation eval = new Evaluation(trainingDataSet);
            //eval.evaluateModel(forest, testingDataSet);
            eval.evaluateModel(ada, testingDataSet);


            /** Print the algorithm summary */
            System.out.println();
            System.out.println(eval.toMatrixString("=== Confusion matrix for fold ==="));
            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("kappa = " + eval.kappa());
            System.out.println("MAE = " + eval.meanAbsoluteError());
            System.out.println("RMSE = " + eval.rootMeanSquaredError());
            System.out.println("RAE = " + eval.relativeAbsoluteError());
            System.out.println("RRSE = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("fMeasure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            //System.out.println(forest);
            System.out.println(ada);
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            e.printStackTrace();
        }

        //public static void main(String args[]) {
        // RandomForestClassifier randomForestClassifier = new RandomForestClassifier();
        // randomForestClassifier.execute();


        //}
    }
}
