using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml.prediction.Data
{
    public class CropData
    {
        [LoadColumn(0)]
        public float Temperature { get; set; }

        [LoadColumn(1)]
        public float Humidity { get; set; }

        [LoadColumn(2)]
        public float Rainfall { get; set; }

        [LoadColumn(3)]
        public float SoilMoisture { get; set; }

        [LoadColumn(4)]
        public float Area { get; set; }

        [LoadColumn(5)]
        public float Yield { get; set; }

    }
}
