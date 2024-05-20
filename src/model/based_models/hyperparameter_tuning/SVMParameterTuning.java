package model.based_models.hyperparameter_tuning;

import weka.classifiers.functions.SMO;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class SVMParameterTuning {

    public static void main(String[] args) {
        try {
            // Load datasets
            Instances trainDataset = loadDataset("data\\segment-challenge.arff");
            Instances testDataset = loadDataset("data\\segment-test.arff");
            Instances validDataset = loadDataset("data\\segment-challenge.arff");

            // Set class index to the last attribute
            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(validDataset);

            // Arrays to hold kernels and their corresponding parameters
            String[] kernels = {
                "weka.classifiers.functions.supportVector.RBFKernel",
                "weka.classifiers.functions.supportVector.PolyKernel",
                "weka.classifiers.functions.supportVector.NormalizedPolyKernel"
            };
            String[] kernelOptions = {
                "-G 0.01", // Options for RBFKernel
                "-E 2.0",  // Options for PolyKernel
                "-E 2.0"   // Options for NormalizedPolyKernel
            };
            double bestAccuracy = -1;
            String[] bestOptions = null;

            // Loop through kernels and perform parameter tuning
            for (int i = 0; i < kernels.length; i++) {
                SMO smo = new SMO();
                smo.setOptions(Utils.splitOptions("-K \"" + kernels[i] + " " + kernelOptions[i] + "\""));

                CVParameterSelection ps = new CVParameterSelection();
                ps.setClassifier(smo);
                ps.setNumFolds(5); // 5-fold cross-validation
                ps.addCVParameter("C 0.1 5.0 10");

                // Perform cross-validation to find the best parameters on the validation dataset
                ps.buildClassifier(validDataset);

                // Create and configure an SMO classifier with the best options
                SMO tempSmo = new SMO();
                tempSmo.setOptions(ps.getBestClassifierOptions());
                tempSmo.buildClassifier(validDataset);

                // Evaluate the model on the validation dataset
                Evaluation eval = new Evaluation(validDataset);
                eval.crossValidateModel(tempSmo, validDataset, 5, new java.util.Random(1));

                // Update the best options if current options are better
                if (eval.pctCorrect() > bestAccuracy) {
                    bestAccuracy = eval.pctCorrect();
                    bestOptions = ps.getBestClassifierOptions();
                }
            }

            // Print the best parameters
            System.out.println("Best Parameters: " + String.join(" ", bestOptions));

            // Train the SMO classifier with the best parameters
            SMO svm = new SMO();
            svm.setOptions(bestOptions);
            svm.buildClassifier(trainDataset);

            // Evaluate the classifier on the test dataset
            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(svm, testDataset);

            // Output the evaluation results
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            // Print the confusion matrix
            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            // Save the model
            SerializationHelper.write("svm.model", svm);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Load a dataset from an ARFF file.
     *
     * @param filePath the path to the ARFF file
     * @return the loaded Instances object
     * @throws Exception if there is an error loading the dataset
     */
    private static Instances loadDataset(String filePath) throws Exception {
        DataSource source = new DataSource(filePath);
        return source.getDataSet();
    }

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
}