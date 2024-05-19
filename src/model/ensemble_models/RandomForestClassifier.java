package model.ensemble_models;


import util.Loader;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestClassifier {
    public void execute() {
        // //Convert train data
        // Loader.loadCsv("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working.csv");
        // Converter.csv2Arff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working.csv",
        // "C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working test.arff");
        // Instances work_data_train = Loader.loadArff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working.arff");

        // //Convert test data
        // Loader.loadCsv("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working test.csv");
        // Converter.csv2Arff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working test.csv",
        // "C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working test.arff");
        // Instances work_data_test = Loader.loadArff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\Working test.arff");

        Instances work_data_train = Loader.loadArff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\segment-challenge.arff");
        Instances work_data_test = Loader.loadArff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\segment-test.arff");
        Instances validSource = Loader.loadArff("C:\\Users\\Lenovo\\Desktop\\Project Data Mining\\data\\segment-challenge.arff");


        try {
            RandomForest forest = new RandomForest();
            


            Evaluation eval = new Evaluation(work_data_train);
            eval.evaluateModel(forest, work_data_test);

            
            // setup classifier
            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(forest);
            ps.setNumFolds(10);  // using 10-fold CV   
            ps.addCVParameter("C 0.1 0.5 5");
            
            forest.buildClassifier(work_data_train);
            System.out.println("=== Random Forest Classifier Model ===\n");
            System.out.println(forest);
            /** Print the algorithm summary */
            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", ps.getBestClassifierOptions()));
            System.out.println("** Decision Tress Evaluation with Datasets **");
            System.out.println(eval.toSummaryString());
            System.out.print(" the expression for the input data as per alogorithm is ");
            System.out.println(forest);
            System.out.println(eval.toMatrixString());
            System.out.println(eval.toClassDetailsString());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
