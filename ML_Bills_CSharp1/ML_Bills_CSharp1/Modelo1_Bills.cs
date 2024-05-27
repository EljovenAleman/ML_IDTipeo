using Microsoft.ML.Data;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_Bills_CSharp1
{
    class Modelo1_Bills
    {
        string _trainingFilePath = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\Modelo 1_Bills.tsv";
        string _modelFilePath = "D:\\Repos\\ML_Bills_CSharp1\\ML_Bills_CSharp1\\Models\\model.zip";
        MLContext _mlContext;
        IDataView _trainingDataView;
        ITransformer _model;
        PredictionEngine<BillItem, TypePrediction> _predictionEngine;

        public void Main()
        {
            _mlContext = new MLContext(seed: 0);
            _trainingDataView = _mlContext.Data.LoadFromTextFile<BillItem>(_trainingFilePath, hasHeader: true);
            var pipeline = ProcessData();
            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
            SaveModelAsFile();

            var result = PredictDeparmentForSubjectLine("6/1243525446");
            var result2 = PredictDeparmentForSubjectLine("22/11/1880");

            Console.ReadLine();
        }
        
        string PredictDeparmentForSubjectLine(string DatoLine)
        {
            var model = _mlContext.Model.Load(_modelFilePath, out var modelInputSchema);
            var billItem = new BillItem() { Dato = DatoLine };
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<BillItem, TypePrediction>(model);
            var result = _predictionEngine.Predict(billItem);
            return result.Tipo;
        }

        void SaveModelAsFile()
        {
            _mlContext.Model.Save(_model, _trainingDataView.Schema, _modelFilePath);
        }

        IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _model = trainingPipeline.Fit(trainingDataView);
            return trainingPipeline;
        }


        IEstimator<ITransformer> ProcessData()
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Tipo", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Dato", outputColumnName: "DatoFeaturized")
                .Append(_mlContext.Transforms.Concatenate("Features", "DatoFeaturized"))
                .AppendCacheCheckpoint(_mlContext));
            return pipeline;
        }


        public class BillItem
        {
            [LoadColumn(0)]
            public string Dato { get; set; }

            [LoadColumn(1)]
            public string Tipo { get; set; }
        }

        public class TypePrediction
        {
            [ColumnName("PredictedLabel")]
            public string? Tipo { get; set; }
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
    }
}
