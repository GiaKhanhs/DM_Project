/* 
import model.ensemble_models.hyperparameter_tuning.AdaBoostM1Tuning;
import model.ensemble_models.hyperparameter_tuning.ExtraTreeTuning;
import model.ensemble_models.hyperparameter_tuning.RandomForestTuning;
import model.ensemble_models.models.AdaBoostM1Classifier;
import model.ensemble_models.models.ExtraTreeClassifier;
import model.ensemble_models.models.RandomForestClassifier;
*/
import model.Command;
import model.based_models.hyperparameter_tuning.*;
import model.based_models.models.*;

public class Main {
    public static void main(String[] args) {
        /* 
        try {
            String trainSource = "data\\segment-challenge.arff";
            String testSource = "data\\segment-test.arff";
            String validSource = "data\\segment-challenge.arff";

            // Call models
            double RF = RandomForestClassifier.Evaluate(trainSource, testSource, validSource);
            double Ada = AdaBoostM1Classifier.Evaluate(trainSource, testSource, validSource);
            double ET = ExtraTreeClassifier.Evaluate(trainSource, testSource, validSource);

            // Call the tuning models
            double RFTuning = RandomForestTuning.tuneAndEvaluate(trainSource, testSource, validSource);
            double AdaTuning = AdaBoostM1Tuning.tuneAndEvaluate(trainSource, testSource, validSource);
            double ETTuning = ExtraTreeTuning.tuneAndEvaluate(trainSource, testSource, validSource);

            // Print the models
            System.out.println("Compare accuracy of each models before tuning");
            System.out.println("Test Accuracy of RandomForest: " + RF);
            System.out.println("Test Accuracy of AdaBoostM1: " + Ada);
            System.out.println("Test Accuracy of ExtraTree " + ET);
            System.out.println("\n");

            // Print the tuning models
            System.out.println("Compare accuracy of each models after tuning");
            System.out.println("Test Accuracy of RandomForest: " + RFTuning);
            System.out.println("Test Accuracy of AdaBoostM1: " + AdaTuning);
            System.out.println("Test Accuracy of ExtraTree: " + ETTuning);

        } catch (Exception e) {
            e.printStackTrace();
        } */
        System.out.println("=============OneR Classification=============");
        OneR();
        System.out.println("=============IBK Classification=============");
        IBk();
        System.out.println("=============Logistic Regression Classification=============");
        LR();
        System.out.println("=============Naive Bayes Classification=============");
        NB();
        System.out.println("=============SVM Classification=============");
        SVM();
        System.out.println("=============J48 Classification=============");
        J48();
    }
    public static void OneR() {
		(new OneRClassifier()).exec();
	}
    public static void IBk() {
		(new IBkClassifier()).exec();
	}
    public static void LR() {
		(new LogisticRegressionClassifier()).exec();
	}
    public static void NB() {
		(new NaiveBayesClassifier()).exec();
	}
    public static void SVM() {
		(new SVMClassifier()).exec();
	}
    public static void J48() {
		(new J48Classifier()).exec();
	}
}
