package util;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

public class Loader {
    public static Instances loadArff(String src) {
        try {
            ArffLoader loader = new ArffLoader();
            loader.setSource(new File(src));
            return loader.getDataSet();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load ARFF file: " + src, e);
        }
    }

    public static Instances loadCsv(String src) {
        try {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(src));
            return loader.getDataSet();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load CSV file: " + src, e);
        }
    }
}