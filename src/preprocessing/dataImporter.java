package preprocessing;

import weka.core.converters.ConverterUtils.DataSource;

public class dataImporter {

    public static DataSource dataSource;
    public static DataSource trainSource;
    public static DataSource testSource;
    public static DataSource validSource;

    static {

        try {
            dataSource = new DataSource("data/family/family.arff");
            trainSource = new DataSource("data/family/training_data.arff");
            testSource = new DataSource("data/family/test_data.arff");
            validSource = new DataSource("data/family/validation_data.arff");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }


}
