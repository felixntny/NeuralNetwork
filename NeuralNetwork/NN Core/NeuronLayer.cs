using System.Collections.Generic;

namespace NeuralNetworks
{
    public class NeuronLayer
    {
        public List<NeuronNode> NeuronNodes { get; set; }
        public int TotalNodes { get; set; }

        public NeuronLayer(int totalNodes, NeuronLayerType neuronLayerType)
        {
            NeuronNodes = new List<NeuronNode>();
            TotalNodes = totalNodes;
            CreateNeuralLayer(neuronLayerType);
        }

        private void CreateNeuralLayer(NeuronLayerType neuronLayerType)
        {
            for (int i = 0; i < TotalNodes; i++)
            {
                if(neuronLayerType != NeuronLayerType.Output &&  i == TotalNodes - 1)
                {
                    // For bias 
                    NeuronNodes.Add(new NeuronNode(NeuronLayerType.Input));
                }
                else
                {
                    NeuronNodes.Add(new NeuronNode(neuronLayerType));

                }
            }
        }
    }
}