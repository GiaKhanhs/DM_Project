package util;

import weka.classifiers.evaluation.Evaluation;

public class Printer {
    public static void printConfusionMatrix(Evaluation eval) throws Exception {
        System.out.println();
        System.out.println(eval.toMatrixString("=== Confusion matrix for fold ===\n"));
        System.out.println("Correct % = " + eval.pctCorrect());
        System.out.println("Precision = " + eval.precision(1));
        System.out.println("Recall = " + eval.recall(1));
        System.out.println("fMeasure = " + eval.fMeasure(1));
        System.out.println("Error Rate = " + eval.errorRate());
        System.out.println();
    }
}