using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    internal static class Program
    {
        private const int InputLayerSize = 784;
        private const int AssociationsLayerSize = 20;
        private const int ResultLayerSize = 10;
        private const int InputLayerLinksSize = InputLayerSize * AssociationsLayerSize;
        private const int AssociationsLayerLinksSize = AssociationsLayerSize * ResultLayerSize;
        private const int TrainRowsCount = 60_000;
        private const int TestRowsCount = 10_000;
        private const string FilePath = @"mnist_train.csv";
        private const string TestPath = @"mnist_test.csv";
        private static readonly double[] LayerInputNodes = new double[InputLayerSize];
        private static readonly double[] LayerAssociationsNodes = new double[AssociationsLayerSize];
        private static readonly double[] LayerAssociationsWeights = new double[InputLayerLinksSize];
        private static readonly double[] LayerAssociationsWeightDeltas = new double[InputLayerLinksSize];
        private static readonly double[] LayerResultNodes = new double[ResultLayerSize];
        private static readonly double[] LayerResultWeights = new double[AssociationsLayerLinksSize];
        private static readonly double[] LayerResultWeightPartialDeltas = new double[AssociationsLayerLinksSize];
        private static readonly double[] LayerResultWeightDeltas = new double[AssociationsLayerLinksSize];
        private const double LearningRate = 0.5;
        private static long _correctResults;
        private static double _error;

        public static void Main()
        {
            InitWeights();
            Train();
            Test();
            PrintAccuracy();
        }

        private static void Train()
        {
            var index = 1;
            var rows = File.ReadAllLines(FilePath).Take(TrainRowsCount).ToList();

            foreach (var row in rows)
            {
                Console.Write($"Iteration {index++} из {TrainRowsCount} ");
                var values = row.Split(',');
                var correctNumber = byte.Parse(values[0]);

                AssignInputNodesLayer(values);
                CalculateAssociationsLayer();
                CalculateResultLayer();
                BackPropagation(correctNumber);
                PrintTotalError(correctNumber);
            }
        }

        private static void Test()
        {
            var index = 1;
            var rows = File.ReadAllLines(TestPath).Take(TestRowsCount).ToList();            

            foreach (var row in rows)
            {
                Console.Write($"Iteration {index++} из {TestRowsCount} ");
                var values = row.Split(',');
                var correctNumber = byte.Parse(values[0]);

                AssignInputNodesLayer(values);
                CalculateAssociationsLayer();
                CalculateResultLayer();
                CalculateStatistics(correctNumber);
            }
        }

        private static void InitWeights()
        {
            var rand = new Random();
            for (var i = 0; i < LayerAssociationsWeights.Length; i++)
                LayerAssociationsWeights[i] = rand.NextDouble() * 0.001;
            for (var i = 0; i < LayerResultWeights.Length; i++)
                LayerResultWeights[i] = rand.NextDouble() * 0.001;
        }

        private static void CalculateStatistics(int correctNumber)
        {
            var max = LayerResultNodes.Max();
            var proposalNumber = 0;
            for (var i = 0; i < LayerResultNodes.Length; i++)
            {
                if (!(Math.Abs(LayerResultNodes[i] - max) < 1E-10)) continue;
                proposalNumber = i;
                break;
            }
            Console.WriteLine("{0}->{1} ({2})", correctNumber, proposalNumber, proposalNumber == correctNumber ? "CORRECT" : "INCORRECT");
            if (proposalNumber == correctNumber) _correctResults++;
        }

        private static void PrintTotalError(int correctNumber)
        {
            _error = 0;
            for (var i = 0; i < LayerResultNodes.Length; i++)
            {
                var target = i == correctNumber ? 1 : 0;
                _error += 0.5 * Math.Pow(target - LayerResultNodes[i], 2);
            }
            Console.WriteLine($"Error: {_error}");
        }

        private static double FunActivation(double value)
        {
            return 1 / (1 + Math.Pow(Math.E, -value));
        }

        private static void AssignInputNodesLayer(IReadOnlyList<string> values)
        {
            for (var i = 1; i < values.Count; i++)
                LayerInputNodes[i - 1] = double.Parse(values[i]) / byte.MaxValue;
        }

        private static void CalculateAssociationsLayer()
        {
            for (var i = 0; i < LayerAssociationsNodes.Length; i++)
            {
                LayerAssociationsNodes[i] = 0;
                for (var j = 0; j < LayerInputNodes.Length; j++)
                    LayerAssociationsNodes[i] += LayerInputNodes[j] * LayerAssociationsWeights[i * LayerInputNodes.Length + j];
                LayerAssociationsNodes[i] = FunActivation(LayerAssociationsNodes[i]);
            }
        }

        private static void CalculateResultLayer()
        {
            for (var i = 0; i < LayerResultNodes.Length; i++)
            {
                LayerResultNodes[i] = 0;
                for (var j = 0; j < LayerAssociationsNodes.Length; j++) 
                {
                    LayerResultNodes[i] += 
                        (LayerAssociationsNodes[j]
                        * LayerResultWeights[i * LayerAssociationsNodes.Length + j]);                                    
                }
                LayerResultNodes[i] = FunActivation(LayerResultNodes[i]);
            }
        }

        private static void BackPropagation(int correctNumber)
        {
            CalculateLayerResultWeightDeltas(correctNumber);
            CalculateLayerAssociationsWeightDeltas();
            UpdateLayerResultWeightDeltas();
            UpdateLayerAssociationsWeightDeltas();
        }

        private static void CalculateLayerResultWeightDeltas(int correctNumber)
        {
            for (var i = 0; i < LayerResultNodes.Length; i++)
            {
                var target = i == correctNumber ? 1 : 0;
                var actual = LayerResultNodes[i];

                for (var j = 0; j < LayerAssociationsNodes.Length; j++)
                {
                    LayerResultWeightPartialDeltas[i * LayerAssociationsNodes.Length + j] =
                        (target - actual) * actual * (1 - actual);
                    LayerResultWeightDeltas[i * LayerAssociationsNodes.Length + j] =
                        LayerResultWeightPartialDeltas[i * LayerAssociationsNodes.Length + j]
                        * LayerAssociationsNodes[j];
                }
            }
        }

        private static double GetSumOutgoingLinks(int index)
        {
            double result = 0;
            for (var i = 0; i < LayerResultNodes.Length; i++)
                result += LayerResultWeightPartialDeltas[LayerAssociationsNodes.Length * i + index] *
                          LayerResultWeights[LayerAssociationsNodes.Length * i + index];
            return result;
        }

        private static void CalculateLayerAssociationsWeightDeltas()
        {
            for (var i = 0; i < LayerAssociationsNodes.Length; i++)
            {
                var sumOutgoingLinks = GetSumOutgoingLinks(i);
                for (var j = 0; j < LayerInputNodes.Length; j++)
                    LayerAssociationsWeightDeltas[i * LayerInputNodes.Length + j] = 
                        sumOutgoingLinks * LayerAssociationsNodes[i] * 
                        (1 - LayerAssociationsNodes[i]) * LayerInputNodes[j];
            }
        }

        private static void UpdateLayerResultWeightDeltas()
        {
            for (var i = 0; i < LayerResultWeights.Length; i++)
                LayerResultWeights[i] += LearningRate * LayerResultWeightDeltas[i];
        }

        private static void UpdateLayerAssociationsWeightDeltas()
        {
            for (var i = 0; i < LayerAssociationsWeights.Length; i++)
                LayerAssociationsWeights[i] += LearningRate * LayerAssociationsWeightDeltas[i];
        }

        private static void PrintAccuracy()
        {
            Console.WriteLine($"Accuracy: {100 * _correctResults / TestRowsCount}%");
        }
    }
}