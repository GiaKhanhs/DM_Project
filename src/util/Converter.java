package util;

import weka.core.Instances;

public class Converter {
    public static void csv2Arff(String src, String dest) {
        Instances data = Loader.loadCsv(src);
        Saver.saveArff(dest, data);
    }

    public static void arff2Csv(String src, String dest) {
        Instances data = Loader.loadArff(src);
        Saver.saveCsv(dest, data);
    }
}