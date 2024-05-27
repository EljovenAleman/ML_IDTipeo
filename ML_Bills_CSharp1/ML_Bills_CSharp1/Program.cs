using iTextSharp.text.pdf.parser;
using iTextSharp.text.pdf;
using Microsoft.ML;
using Microsoft.ML.Data;


string _trainingFilePath = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\TrainingData_4.tsv";

string _modelFilePath = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\TM_IDCargos_2.zip";
string _modelFilePath2 = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\TM_IDCargos_2.zip";
string _modelFilePath3 = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\TM_IDCargos_2.zip";

MLContext _mlContext;
IDataView _trainingDataView;
ITransformer _model;
PredictionEngine<InputData, IdCargoPrediction> _predictionEngine;

_mlContext = new MLContext(seed: 0);
_trainingDataView = _mlContext.Data.LoadFromTextFile<InputData>(_trainingFilePath,hasHeader: true);

var pipelineIDCargo = ProcessDataIDCargo();
var IdCargoModel = BuildAndTrainModel(_trainingDataView, pipelineIDCargo);

var pipelineIDTipo = ProcessDataIDTipo();
var IdTipoModel = BuildAndTrainModel(_trainingDataView, pipelineIDTipo);

var pipelineIDTipo2 = ProcessDataIDTipo2();
var IdTipo2Model = BuildAndTrainModel(_trainingDataView, pipelineIDTipo2);

//SaveModelAsFile();

var IdCargoPredictor = _mlContext.Model.CreatePredictionEngine<InputData, IdCargoPrediction>(IdCargoModel);
var IdTipoPredictor = _mlContext.Model.CreatePredictionEngine<InputData, IdTipoPrediction>(IdTipoModel);
var IdTipo2Predictor = _mlContext.Model.CreatePredictionEngine<InputData, IdTipo2Prediction>(IdTipo2Model);

var newData = new InputData { CargoOK = "Abono Roaming 600Mb Mundial" };

var Predicted_IdCargo = IdCargoPredictor.Predict(newData);
newData.IdCargo = Predicted_IdCargo.IdCargo;

var Predicted_IdTipo = IdTipoPredictor.Predict(newData);
newData.IdTipo = Predicted_IdTipo.IdTipo;

var Predicted_IdTipo2 = IdTipo2Predictor.Predict(newData);
newData.IdTipo2 = Predicted_IdTipo2.IdTipo2;



Console.ReadLine();








string GetPrediction(string DatoLine)
{
    var model = _mlContext.Model.Load(_modelFilePath, out var modelInputSchema);
    var billItem = new InputData() { CargoOK = DatoLine };
    _predictionEngine = _mlContext.Model.CreatePredictionEngine<InputData, IdCargoPrediction>(model);
    var result = _predictionEngine.Predict(billItem);
    return result.IdCargo;
}

void SaveModelAsFile()
{
    _mlContext.Model.Save(_model, _trainingDataView.Schema, _modelFilePath);
}

ITransformer BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    return _model = trainingPipeline.Fit(trainingDataView);    
}


IEstimator<ITransformer> ProcessDataIDCargo()
{
    var textpipeline = _mlContext.Transforms.Text.FeaturizeText("Features", "CargoOK");

    var pipeline = textpipeline
        .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "IdCargo"))      
        .Append(_mlContext.Transforms.Concatenate("Features", "Features"))
        .Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(_mlContext.BinaryClassification.Trainers.AveragedPerceptron(), labelColumnName: "Label"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue(nameof(IdCargoPrediction.IdCargo), "Label"))
        .AppendCacheCheckpoint(_mlContext);
    return pipeline;
}

IEstimator<ITransformer> ProcessDataIDTipo()
{
    var textpipeline = _mlContext.Transforms.Text.FeaturizeText("Features", "CargoOK");

    var pipeline = textpipeline
        .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "IdTipo"))
        .Append(_mlContext.Transforms.Concatenate("Features", "Features"))
        .Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(_mlContext.BinaryClassification.Trainers.AveragedPerceptron(), labelColumnName: "Label"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue(nameof(IdCargoPrediction.IdCargo), "Label"))
        .AppendCacheCheckpoint(_mlContext);
    return pipeline;
}

IEstimator<ITransformer> ProcessDataIDTipo2()
{
    var textpipeline = _mlContext.Transforms.Text.FeaturizeText("Features", "CargoOK");

    var pipeline = textpipeline
        .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label", "IdTipo2"))
        .Append(_mlContext.Transforms.Concatenate("Features", "Features"))
        .Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(_mlContext.BinaryClassification.Trainers.AveragedPerceptron(), labelColumnName: "Label"))
        .Append(_mlContext.Transforms.Conversion.MapKeyToValue(nameof(IdCargoPrediction.IdCargo), "Label"))
        .AppendCacheCheckpoint(_mlContext);
    return pipeline;
}


public class InputData
{
    [LoadColumn(0)]
    public string CargoOK { get; set; }

    [LoadColumn(1)]
    public string IdCargo { get; set; }

    [LoadColumn(2)]
    public string IdTipo { get; set; }

    [LoadColumn(3)]
    public string IdTipo2 { get; set; }    
}

public class IdCargoPrediction
{
    [ColumnName("PredictedLabel")]
    public string? IdCargo { get; set; }
}

public class IdTipoPrediction
{
    [ColumnName("PredictedLabel")]
    public string? IdTipo { get; set; }
}

public class IdTipo2Prediction
{
    [ColumnName("PredictedLabel")]
    public string? IdTipo2 { get; set; }
}


//string[] lineas;
//    //// Ruta al archivo PDF
//    //string rutaArchivo = "C:\\Users\\diego\\OneDrive\\Escritorio\\MovilClaro\\6807_1039317969.pdf";
//    //
//    //// Abre el archivo PDF
//    //using (PdfReader reader = new PdfReader(rutaArchivo))
//    //{
//    //    // Itera sobre cada página del PDF
//    //    for (int pagina = 1; pagina <= reader.NumberOfPages; pagina++)
//    //    {
//    //        // Obtiene el texto de la página actual
//    //        string textoPagina = PdfTextExtractor.GetTextFromPage(reader, pagina);
//    //
//    //        // Divide el texto en líneas
//    //        string[] aux = textoPagina.Split('\n');
//    //
//    //        // Itera sobre cada línea e imprímela
//    //        foreach (string linea in aux)
//    //        {
//    //            Console.WriteLine(linea);
//    //        }
//    //    }
//    //}