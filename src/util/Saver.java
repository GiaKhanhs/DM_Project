package util;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.io.IOException;

public class Saver {
    public static void saveArff(String dest, Instances data) {
        try {
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(dest));
            saver.writeBatch();
        } catch (IOException e) {
            throw new RuntimeException("Failed to save ARFF file: " + dest, e);
        }
    }

    public static void saveCsv(String dest, Instances data) {
        try {
            CSVSaver saver = new CSVSaver();
            saver.setInstances(data);
            saver.setFile(new File(dest));
            saver.writeBatch();
        } catch (IOException e) {
            System.out.println(e.getStackTrace());
            throw new RuntimeException("Failed to save CSV file: " + dest, e);
        }
    }
}