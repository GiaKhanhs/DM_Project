package preprocessing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;
import java.util.Random;

import static preprocessing.dataImporter.dataSource;

public class SplitData {
    public static void splitData() throws Exception {
        // Load dataset
        Instances allData = dataSource.getDataSet();

        // Randomize the dataset with a seed
        allData.randomize(new Random(42));

        // Calculate indices for splitting
        int totalInstances = allData.numInstances();
        int testSize = (int) Math.round(totalInstances * 0.2);
        int trainSize = totalInstances - testSize;
        int validationSize = (int) Math.round(trainSize * 0.2);
        trainSize -= validationSize;

        // Create empty sets for train, test, and validation
        Instances testData = new Instances(allData, 0, testSize);
        Instances validationData = new Instances(allData, testSize, validationSize);
        Instances trainData = new Instances(allData, testSize + validationSize, trainSize);

        // Save the datasets to ARFF files
        saveInstances(trainData, "training_data.arff");
        saveInstances(testData, "test_data.arff");
        saveInstances(validationData, "validation_data.arff");
    }

    private static void saveInstances(Instances data, String fileName) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(fileName));
        saver.writeBatch();
    }
}