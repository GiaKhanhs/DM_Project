import model.based_models.hyperparameter_tuning.J48Tuning;
import model.based_models.hyperparameter_tuning.LogisticRegressionTuning;
import model.based_models.hyperparameter_tuning.SVMParameterTuning;
import model.based_models.models.*;
import model.ensemble_models.hyperparameter_tuning.AdaBoostM1Tuning;
import model.ensemble_models.hyperparameter_tuning.ExtraTreeTuning;
import model.ensemble_models.hyperparameter_tuning.RandomForestTuning;
import model.ensemble_models.models.AdaBoostM1Classifier;
import model.ensemble_models.models.ExtraTreeClassifier;
import model.ensemble_models.models.RandomForestClassifier;
import preprocessing.dataImporter;


public class Main {
    public static void main(String[] args) {

        // RandomForest
        System.out.println("=============RandomForest Classification=============");
        RandomForest();

        System.out.println("=============RandomForestTuning Classification=============");
        RandomForestTuning();

        // // AdaBoostM1
        System.out.println("=============AdaBoostM1 Classification=============");
        AdaBoostM1();

        System.out.println("=============AdaBoostM1Tuning Classification=============");
        AdaBoostM1Tuning();

        // ExtraTree
        System.out.println("=============ExtraTree Classification=============");
        ExtraTree();

        System.out.println("=============ExtraTreeTuning Classification=============");
        ExtraTreeTuning();

        // OneR
        System.out.println("=============OneR Classification=============");
        OneR();

        // IBk
        System.out.println("=============IBK Classification=============");
        IBk();

        // Naive Bayes
        System.out.println("=============Naive Bayes Classification=============");
        NB();

        // Logistic Regression
        System.out.println("=============Logistic Regression Classification=============");
        LR();
        System.out.println("=============Logistic Regression Tuning=============");
        LRTuning();

        // SVM
        System.out.println("=============SVM Classification=============");
        SVM();
        System.out.println("=============SVM Tuning=============");
        SVMTuning();

        //J48
        System.out.println("=============J48 Classification=============");
        J48();
        System.out.println("=============J48 Tuning=============");
        J48Tuning();

    }


    public static void RandomForest() {
        (new RandomForestClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void RandomForestTuning() {
        (new RandomForestTuning()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void ExtraTree() {
        (new ExtraTreeClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void ExtraTreeTuning() {
        (new ExtraTreeTuning()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void AdaBoostM1() {
        (new AdaBoostM1Classifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void AdaBoostM1Tuning() {
        (new AdaBoostM1Tuning()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void OneR() {
        (new OneRClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void IBk() {
        (new IBkClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void LR() {
        (new LogisticRegressionClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void NB() {
        (new NaiveBayesClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void SVM() {
        (new SVMClassifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void J48() {
        (new J48Classifier()).exec(dataImporter.trainSource, dataImporter.testSource);
    }

    public static void LRTuning() {
        (new LogisticRegressionTuning()).exec();
    }

    public static void J48Tuning() {
        (new J48Tuning()).exec();
    }

    public static void SVMTuning() {
        (new SVMParameterTuning()).exec();
    }
}
