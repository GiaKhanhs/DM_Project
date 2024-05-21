package preprocessing;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;

import java.io.FileWriter;
import java.io.PrintWriter;

public class featureSelection {
    public static void selectFeatures() throws Exception {
        // Load dataset
        DataSource source = new DataSource("data\\family\\family.arff");
        Instances data = source.getDataSet();

        // Set class index if the class index isn't set yet in the data
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // 1. CfsSubsetEval with GreedyStepwise
        AttributeSelection attSelectCfs = new AttributeSelection();
        CfsSubsetEval evalCfs = new CfsSubsetEval();
        GreedyStepwise searchCfs = new GreedyStepwise();
        searchCfs.setSearchBackwards(false);
        attSelectCfs.setEvaluator(evalCfs);
        attSelectCfs.setSearch(searchCfs);
        attSelectCfs.SelectAttributes(data);
        int[] indicesCfs = attSelectCfs.selectedAttributes();

        // Save selected attributes for CfsSubsetEval to file
        try (PrintWriter writer = new PrintWriter(new FileWriter("selected_attributes_cfs.txt"))) {
            writer.println("Selected attributes by CfsSubsetEval and GreedyStepwise:");
            for (int index : indicesCfs) {
                writer.println(data.attribute(index).name());
            }
        }

        // 2. ReliefFAttributeEval with Ranker
        AttributeSelection attSelectReliefF = new AttributeSelection();
        ReliefFAttributeEval evalReliefF = new ReliefFAttributeEval();
        Ranker searchReliefF = new Ranker();
        attSelectReliefF.setEvaluator(evalReliefF);
        attSelectReliefF.setSearch(searchReliefF);
        attSelectReliefF.SelectAttributes(data);
        int[] indicesReliefF = attSelectReliefF.selectedAttributes();

        // Save selected attributes for ReliefF to file
        try (PrintWriter writer = new PrintWriter(new FileWriter("selected_attributes_reliefF.txt"))) {
            writer.println("Selected attributes by ReliefFAttributeEval and Ranker:");
            for (int index : indicesReliefF) {
                writer.println(data.attribute(index).name());
            }
        }
        System.out.println("Feature selection output files have been saved.");
    }
}