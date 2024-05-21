public class dataProcessor {
    public static void main(String[] args) {
        try {
            // Perform feature selection
            featureSelection.selectFeatures();
            
            // Remove unwanted attributes
            RemoveAttributes.removeAttributes();
            
            // Split the data into training, testing, and validation sets
            SplitData.splitData();
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
