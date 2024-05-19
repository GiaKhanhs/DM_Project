package model.ensemble_models;


import java.io.File;
import java.io.IOException;

import util.Loader;
import util.Converter;
import util.Printer;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;

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

        try {
            RandomForest forest = new RandomForest();
            forest.buildClassifier(work_data_train);


            Evaluation eval = new Evaluation(work_data_train);
		    eval.evaluateModel(forest, work_data_test);

            System.out.println("=== Random Forest Classifier Model ===\n");
			System.out.println(forest);
			/** Print the algorithm summary */
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
