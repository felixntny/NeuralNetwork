using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworks
{
    class Program
    {
        public static NeuralNetwork network = null;
        static void Main(string[] args)
        {
            network = new NeuralNetwork(3, 3, 3, 1);


            FeedData();
            Console.ReadLine();
        }

        private static void FeedData()
        {
            int iterationStatus = 0;
            int index = 0;


            for (int i = 0; i < 1000000; i++)
            {
                decimal randomNumber1 = 1;
                decimal randomNumber2 = 1;
                // decimal randomNumber1 = new Random().Next(0, 100000000);
                // decimal randomNumber2 = randomNumber1 + i;

                // decimal exp = (randomNumber1 + randomNumber2)/ 100000000;

                //randomNumber1 = randomNumber1 / 100000000;
                // randomNumber2 = randomNumber2 / 100000000;

                if (iterationStatus == 0)
                {
                    randomNumber1 = 1;
                    randomNumber2 = 1;
                    iterationStatus++;
                }
                else if (iterationStatus == 1)
                {
                    randomNumber1 = 1;
                    randomNumber2 = 0;
                    iterationStatus++;
                }
                else if (iterationStatus == 2)
                {
                    randomNumber1 = 0;
                    randomNumber2 = 1;
                    iterationStatus++;
                }
                else
                {
                    randomNumber1 = 0;
                    randomNumber2 = 0;
                    iterationStatus = 0;
                }

                decimal exp = (randomNumber1 == 1 || randomNumber2 == 1 ? 1 : 0);

                var dataa =    trainingDataForFlowerColor(index);
                randomNumber1 = dataa[0, 0];
                randomNumber2 = dataa[0, 1];
                exp = dataa[0, 2];
                index++;
                if(index == 8)
                {
                    index = 0;
                }
                //      int randomNumber2 = randomNumber1 == 1 ? 0 : 1;// new Random().Next(0,2);
                //           Console.WriteLine(randomNumber1 + " " + randomNumber2 );


                List<NeuronNode> neuronNodes = new List<NeuronNode>();
                neuronNodes.Add(new NeuronNode(NeuronLayerType.Input) { NetValue = (decimal)randomNumber1 });
                neuronNodes.Add(new NeuronNode(NeuronLayerType.Input) { NetValue = (decimal)randomNumber2 });
                neuronNodes.Add(new NeuronNode(NeuronLayerType.Input) { NetValue = 1 });

                decimal[] expectedValues = { exp };

                network.AddInputValues(neuronNodes);
                network.AddExpectedValues(expectedValues);

                network.CopyModifiedWeightMathrixFromWeightMatrix();
                network.CalculateNetValues();
                network.CalculateLayerErrorValues();
                network.RunBackPropagation();
                network.CopyModifiedWeightMathrix();

                Console.Write(Math.Round(network.TotalErrorValue, 8) + " ");

                if (network.TotalErrorValue < 0.0000001m)
                {
                    //           break;
                }
            }


            Console.Write(Math.Round(network.TotalErrorValue, 3) + " ");



        }

        private static decimal[,] trainingDataForFlowerColor(int index)
        {
            decimal[,] data = { {3, 1.5M, 1 },
                                { 2,   1,   0 },
                                { 4,   1.5M, 1 },
                                { 3,   1,   0 },
                                { 3.5M, .5M,  1 },
                                { 2,   .5M,  0 },
                                { 5.5M,  1,  1},
                                {1,    1,  0}
                                 } ;

            var returnValue =  new decimal[1,3]  ;
            returnValue[0, 0] = data[index, 0];
            returnValue[0, 1] = data[index, 1];
            returnValue[0, 2] = data[index, 2];
            return returnValue;
        }
    }
}
