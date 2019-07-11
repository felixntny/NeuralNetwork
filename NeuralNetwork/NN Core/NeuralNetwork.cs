using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class NeuralNetwork
    {
        private readonly int _numberOfHiddenLayers;
        private readonly int _numberOfNodeInEachHiddenLayers;
        private readonly int _numberOfNodeInInputLayers;
        private readonly int _numberOfNodeInOutputLayers;
        private readonly List<NeuronLayer> _neuronLayer;
     //   private const decimal eta = 0.000001m;
        private const decimal eta = 0.005m;


        private  decimal[] _outputExpectedValueLayer;
        private List<decimal[,]> weightMatrix;



        private List<decimal[,]> ModifiedWeightMatrix;

        public decimal TotalErrorValue { get; set; }

        public NeuralNetwork(int numberOfHiddenLayers, int numberOfNodeInEachHiddenLayers, int numberOfNodeInInputLayers, int numberOfNodeInOutputLayers)
        {
            _numberOfHiddenLayers = numberOfHiddenLayers;
            _numberOfNodeInEachHiddenLayers = numberOfNodeInEachHiddenLayers;
            _numberOfNodeInInputLayers = numberOfNodeInInputLayers;
            _numberOfNodeInOutputLayers = numberOfNodeInOutputLayers;
            _neuronLayer = new List<NeuronLayer>();
            _outputExpectedValueLayer = new decimal[_numberOfNodeInOutputLayers];

            ConstructNeuralNetwork();
            CreateWeights();
            AddDefaultInputValues();
            FeedExpectedValues();
            AddDefaultWeights();
            CopyModifiedWeightMathrixFromWeightMatrix();
            CalculateNetValues();
            CalculateLayerErrorValues();
            RunBackPropagation();
            CopyModifiedWeightMathrix();
        }

        public void ConstructNeuralNetwork()
        {
            for (int i = 0; i < _numberOfHiddenLayers; i++)
            {
                if (i == 0)
                {
                    _neuronLayer.Add(new NeuronLayer(_numberOfNodeInInputLayers, NeuronLayerType.Input));
                }
                else if (i == _numberOfHiddenLayers - 1)
                {
                    _neuronLayer.Add(new NeuronLayer(_numberOfNodeInOutputLayers, NeuronLayerType.Output));
                }
                else
                {
                    _neuronLayer.Add(new NeuronLayer(_numberOfNodeInEachHiddenLayers, NeuronLayerType.Hidden));
                }
            }


        }

        public void CreateWeights()
        {
            weightMatrix = new List<decimal[,]>();
            for (int layerIndex = 1; layerIndex < _numberOfHiddenLayers; layerIndex++)
            {
                {
                    int nodeCount = _neuronLayer.ElementAt(layerIndex).TotalNodes - 1;
                    if (layerIndex == _numberOfHiddenLayers - 1)
                    {
                        nodeCount++;
                    }

                    int prevLayerNodeCount = _neuronLayer.ElementAt(layerIndex-1).TotalNodes;
                    weightMatrix.Add(new decimal[prevLayerNodeCount, nodeCount]);
                }
            }




        }

        private void AddDefaultInputValues()
        {
            decimal initValue = 0.05m;
            for (int inputNodeIndex = 0; inputNodeIndex < _neuronLayer.ElementAt(0).NeuronNodes.Count - 1; inputNodeIndex++)
            {
                _neuronLayer.ElementAt(0).NeuronNodes.ElementAt(inputNodeIndex).NetValue = initValue;
                initValue += 0.05m;
            }

            // Set BIAS value as 1
            // _numberOfHiddenLayers-1  for ignoring output layer
            for (int layerIndex = 0; layerIndex < _numberOfHiddenLayers - 1; layerIndex++)
            {
                _neuronLayer.ElementAt(layerIndex).NeuronNodes.ElementAt(_neuronLayer.ElementAt(layerIndex).NeuronNodes.Count - 1).NetValue = 1;
            }


        }

        internal void AddInputValues(List<NeuronNode> inputLayerNodes)
        {
            _neuronLayer.ElementAt(0).NeuronNodes = inputLayerNodes;
        }

        public void AddExpectedValues(decimal[] expectedValues)
        {
            _outputExpectedValueLayer = expectedValues;
        }


        public void FeedExpectedValues()
        {
            _outputExpectedValueLayer[0] = 0.01m;
           //_outputExpectedValueLayer[1] = 0.99m;
        }

        public void AddDefaultWeights()
        {
            // default weight
            decimal initValue = 0.15m;
            for (int layerIndex = 1; layerIndex < _numberOfHiddenLayers; layerIndex++)
            {
                int nodeCount = _neuronLayer.ElementAt(layerIndex).NeuronNodes.Count - 1;
                if (layerIndex == _numberOfHiddenLayers - 1)
                {
                    nodeCount++;
                }
                int prevLayerNodeCount = _neuronLayer.ElementAt(layerIndex - 1).NeuronNodes.Count;

                for (int prevLayerNodeIndex = 0; prevLayerNodeIndex < prevLayerNodeCount; prevLayerNodeIndex++)
                {
                    for (int currentLayerNodeIndex = 0; currentLayerNodeIndex < nodeCount; currentLayerNodeIndex++)
                    {
                        var data = weightMatrix.ElementAt(layerIndex - 1);
                        data[prevLayerNodeIndex, currentLayerNodeIndex] = initValue;
                        initValue += 0.05m;
                    }
                }
            }

            weightMatrix.ElementAt(0)[0, 0] = 0.15m;
            weightMatrix.ElementAt(0)[1, 0] = 0.2m;
            weightMatrix.ElementAt(0)[0, 1] = 0.25m;
            weightMatrix.ElementAt(0)[1, 1] = 0.3m;
            weightMatrix.ElementAt(0)[2, 0] = 0.35m;
            weightMatrix.ElementAt(0)[2, 1] = 0.35m;
            weightMatrix.ElementAt(1)[0, 0] = 0.4m;
            weightMatrix.ElementAt(1)[1, 0] = 0.45m;
      //      weightMatrix.ElementAt(1)[0, 1] = 0.5m;
      //      weightMatrix.ElementAt(1)[1, 1] = 0.55m;
            weightMatrix.ElementAt(1)[2, 0] = 0.6m;
      //      weightMatrix.ElementAt(1)[2, 1] = 0.6m;


            ModifiedWeightMatrix = weightMatrix;


        }

        internal void CalculateNetValues()
        {
            for (int layerIndex = 1; layerIndex < _numberOfHiddenLayers; layerIndex++)
            {
                int hiddenNodeCount = _neuronLayer.ElementAt(layerIndex).TotalNodes - 1;
                if (layerIndex == _numberOfHiddenLayers - 1)
                {
                    hiddenNodeCount++;
                }
                for (int hiddenLayerNodeIndex = 0; hiddenLayerNodeIndex < hiddenNodeCount; hiddenLayerNodeIndex++)
                {
                    decimal netValue = 0;
                    for (int previousLayerNodeIndex = 0; previousLayerNodeIndex < _neuronLayer.ElementAt(layerIndex - 1).TotalNodes; previousLayerNodeIndex++)
                    {
                        netValue += (_neuronLayer.ElementAt(layerIndex - 1).NeuronNodes.ElementAt(previousLayerNodeIndex).OutputValue * weightMatrix.ElementAt(layerIndex - 1)[previousLayerNodeIndex, hiddenLayerNodeIndex]);
                    }
                    _neuronLayer.ElementAt(layerIndex).NeuronNodes.ElementAt(hiddenLayerNodeIndex).NetValue = netValue;

                }
            }
        }

        internal void CalculateLayerErrorValues()
        {
            decimal totalErrors = 0m;
            for (int outputLayerIndex = 0; outputLayerIndex < _numberOfNodeInOutputLayers; outputLayerIndex++)
            {
                double diff = (double)(_outputExpectedValueLayer[outputLayerIndex] - _neuronLayer.ElementAt(_numberOfHiddenLayers - 1).NeuronNodes.ElementAt(outputLayerIndex).OutputValue);
                var errorValue = Math.Pow((diff), 2) / 2.0;
                totalErrors += ((decimal)errorValue);
                _neuronLayer.ElementAt(_numberOfHiddenLayers - 1).NeuronNodes.ElementAt(outputLayerIndex).ErrorValue = (decimal)errorValue;

            }
            TotalErrorValue = totalErrors;
            CalculateHiddenLayerError();
        }
        private void CalculateHiddenLayerError()
        {

            for (int outputLayerNodeIndex = 0; outputLayerNodeIndex < _numberOfNodeInOutputLayers; outputLayerNodeIndex++)
            {
                NeuronNode outputNode = _neuronLayer.ElementAt(_numberOfHiddenLayers - 1).NeuronNodes.ElementAt(outputLayerNodeIndex);
                decimal delta =  (_outputExpectedValueLayer[outputLayerNodeIndex] - outputNode.OutputValue) * outputNode.OutputValue * (1 - outputNode.OutputValue);
                outputNode.ErrorValue = delta;
            }


            for (int hiddenLayerIndex = _numberOfHiddenLayers - 2; hiddenLayerIndex > 0; hiddenLayerIndex--)
            {
                int lastHiddenLayerTotalNodeCount = _neuronLayer.ElementAt(hiddenLayerIndex).TotalNodes ;
                for (int hiddenLayerNodeIndex = 0; hiddenLayerNodeIndex < lastHiddenLayerTotalNodeCount; hiddenLayerNodeIndex++)
                {
                    NeuronNode Node = _neuronLayer.ElementAt(hiddenLayerIndex).NeuronNodes.ElementAt(hiddenLayerNodeIndex);

                    int nextHiddenLayerTotalNodeCount = _neuronLayer.ElementAt(hiddenLayerIndex + 1).TotalNodes-1;
                    if(hiddenLayerIndex + 1 == _numberOfHiddenLayers - 1)
                    {
                        nextHiddenLayerTotalNodeCount = nextHiddenLayerTotalNodeCount + 1;
                    }
                    decimal nodeError = 0m;
                    for (int nextHiddenLayerNodeIndex = 0; nextHiddenLayerNodeIndex < nextHiddenLayerTotalNodeCount; nextHiddenLayerNodeIndex++)
                    {
                        NeuronNode nextNode = _neuronLayer.ElementAt(hiddenLayerIndex + 1).NeuronNodes.ElementAt(nextHiddenLayerNodeIndex);
                        decimal weightValue = weightMatrix.ElementAt(hiddenLayerIndex)[hiddenLayerNodeIndex, nextHiddenLayerNodeIndex];
                        nodeError += (nextNode.ErrorValue * weightValue);
                    }

                    Node.ErrorValue = nodeError * (Node.OutputValue * (1- Node.OutputValue));
                }
            }
        }

        internal void RunBackPropagation()
        {
            int lastHiddenLayerTotalNodeCount = _neuronLayer.ElementAt(_numberOfHiddenLayers - 2).TotalNodes;
            for (int lastHiddenLayerNodeIndex = 0; lastHiddenLayerNodeIndex < lastHiddenLayerTotalNodeCount; lastHiddenLayerNodeIndex++)
            {
                    NeuronNode prevNode = _neuronLayer.ElementAt(_numberOfHiddenLayers - 2).NeuronNodes.ElementAt(lastHiddenLayerNodeIndex);
                for (int outputLayerIndex = 0; outputLayerIndex < _numberOfNodeInOutputLayers; outputLayerIndex++)
                {
                    NeuronNode outputNode = _neuronLayer.ElementAt(_numberOfHiddenLayers - 1).NeuronNodes.ElementAt(outputLayerIndex);
                    decimal delta =  (_outputExpectedValueLayer[outputLayerIndex] - outputNode.OutputValue) * (outputNode.OutputValue * (1 - outputNode.OutputValue));
                    ModifiedWeightMatrix.ElementAt(_numberOfHiddenLayers - 2)[lastHiddenLayerNodeIndex, outputLayerIndex] = (weightMatrix.ElementAt(_numberOfHiddenLayers - 2)[lastHiddenLayerNodeIndex, outputLayerIndex] + (eta * delta * prevNode.OutputValue));
                }
            }

            for (int hiddenLayerIndex = _numberOfHiddenLayers - 3; hiddenLayerIndex >= 0; hiddenLayerIndex--)
            {
                int hiddenLayerTotalNodeCount = _neuronLayer.ElementAt(hiddenLayerIndex).TotalNodes;
                for (int hiddenLayerNodeIndex = 0; hiddenLayerNodeIndex < hiddenLayerTotalNodeCount; hiddenLayerNodeIndex++)
                {
                    NeuronNode Node = _neuronLayer.ElementAt(hiddenLayerIndex).NeuronNodes.ElementAt(hiddenLayerNodeIndex);

                    int nextHiddenLayerTotalNodeCount = _neuronLayer.ElementAt(hiddenLayerIndex + 1).TotalNodes - 1;
                    for (int nextHiddenLayerNodeIndex = 0; nextHiddenLayerNodeIndex < nextHiddenLayerTotalNodeCount; nextHiddenLayerNodeIndex++)
                    {
                        NeuronNode nextNode = _neuronLayer.ElementAt(hiddenLayerIndex + 1).NeuronNodes.ElementAt(nextHiddenLayerNodeIndex);
                        decimal delta = nextNode.ErrorValue ;
                        ModifiedWeightMatrix.ElementAt(hiddenLayerIndex)[hiddenLayerNodeIndex, nextHiddenLayerNodeIndex] = (weightMatrix.ElementAt(hiddenLayerIndex)[hiddenLayerNodeIndex, nextHiddenLayerNodeIndex] + (eta * delta * Node.OutputValue));
                    }

                }
            }
        }


        internal void CopyModifiedWeightMathrix()
        {
            weightMatrix = ModifiedWeightMatrix;
        }

        internal void CopyModifiedWeightMathrixFromWeightMatrix()
        {
            ModifiedWeightMatrix = weightMatrix;
        }
    }


}