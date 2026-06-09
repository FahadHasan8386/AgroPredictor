using Microsoft.ML;
using Ml.prediction.Data;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        //Load data
        IDataView data = mlContext.Data.LoadFromTextFile<CropData>(
            path: @"E:\Repository\AgroPredictor\crop-data.csv",
            hasHeader: false,
            separatorChar: ',');

        // Pipeline
        var pipeline =
            mlContext.Transforms.Concatenate(
                "Features",
                nameof(CropData.Temperature),
                nameof(CropData.Humidity),
                nameof(CropData.Rainfall),
                nameof(CropData.SoilMoisture),
                nameof(CropData.Area))
            .Append(
                mlContext.Regression.Trainers.Sdca(
                    labelColumnName: "Yield",
                    featureColumnName: "Features"));

        Console.WriteLine("Training Model...");

        var model = pipeline.Fit(data);

        Console.WriteLine("Training Completed!");

        var predictor =
            mlContext.Model.CreatePredictionEngine<CropData, CropPrediction>(model);

        // User Input
        Console.Write("Temperature: ");
        float temp = float.Parse(Console.ReadLine());

        Console.Write("Humidity: ");
        float humidity = float.Parse(Console.ReadLine());

        Console.Write("Rainfall: ");
        float rainfall = float.Parse(Console.ReadLine());

        Console.Write("Soil Moisture: ");
        float soil = float.Parse(Console.ReadLine());

        Console.Write("Area (Acre): ");
        float area = float.Parse(Console.ReadLine());

        var sample = new CropData
        {
            Temperature = temp,
            Humidity = humidity,
            Rainfall = rainfall,
            SoilMoisture = soil,
            Area = area
        };

        var predictions = model.Transform(data);

        var metrics = mlContext.Regression.Evaluate(
            predictions,
            labelColumnName: "Yield");

        Console.WriteLine($"R² Score: {metrics.RSquared}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");

        var result = predictor.Predict(sample);

        Console.WriteLine();
        Console.WriteLine($"Predicted Yield = {result.PredictedYield:F2} KG");
    }
}