using System;

namespace NeuralNetworks
{
    public class NeuronNode
    {
        public decimal NetValue { get; set; }
        public decimal ErrorValue
        {
            get; set;
        }
        public decimal OutputValue { get { return GetOutputValue(); } }
        private readonly NeuronLayerType _neuronNodeType;

        public NeuronNode(NeuronLayerType neuronLayerType)
        {
            _neuronNodeType = neuronLayerType;
        }
        public decimal GetOutputValue()
        {
            if (_neuronNodeType != NeuronLayerType.Input)
            {
                decimal exponentValue =(decimal) Math.Pow(2.7182818284590452353602875, (double)NetValue * -1);
                decimal _outputValue = 1 / (1 + exponentValue);
                _outputValue = Math.Round(_outputValue, 9);
                return _outputValue;
            }
            else
            {
                return NetValue;
            }

        }
    }
}