package preprocessing;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileWriter;
import java.io.PrintWriter;

public class RemoveAttributes {
    public static void removeAttributes() throws Exception {
        // Load dataset
        DataSource source = new DataSource("data\\family\\family.arff");
        Instances data = source.getDataSet();

        // Setup filter
        Remove remove = new Remove();
        remove.setAttributeIndices("1-4, 7, 8, 10-12, 14, 16, 18-22, 24, 25");   //Remove attributes of Family
            //remove.setAttributeIndices("1-4, 6-11, 13-18, 21-25");  Remove attributes of Work
            //remove.setAttributeIndices("1, 2, 4, 5, 7-10, 12, 13, 17, 19-25"); Remove attributes of Love
            //remove.setAttributeIndices("1-5, 7-10, 12-17, 19-23, 25");  Remove attributes of Self
            //remove.setAttributeIndices("1-4, 8-10, 12-20, 22, 24, 25");  //Remove attributes of Study
        remove.setInputFormat(data);

        // Apply filter
        Instances newData = Filter.useFilter(data, remove);

        // Save new dataset
        try (PrintWriter writer = new PrintWriter(new FileWriter("data_FS.arff"))) {
            writer.println(newData);
        }
    }
}