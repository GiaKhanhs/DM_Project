package model.ensemble_models.models;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AdaBoostM1Classifier {
    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }

    public static double Evaluate(String trainPath, String testPath, String validPath) throws Exception {
        
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

        AdaBoostM1 ada = new AdaBoostM1();
        ada.buildClassifier(trainingDataSet);

        Evaluation eval = new Evaluation(trainingDataSet);
        eval.evaluateModel(ada, testingDataSet);


        System.out.println("=== AdaBoostM1 Classifier Model ===\n");
        System.out.println(ada);
        /** Print the algorithm summary */
        System.out.println("** Decision Tress Evaluation with Datasets **");
        System.out.println(eval.toSummaryString());
        System.out.print(" the expression for the input data as per alogorithm is ");
        System.out.println(ada);
        System.out.println(eval.toMatrixString());
        System.out.println(eval.toClassDetailsString());

        return eval.pctCorrect();
    }
}
