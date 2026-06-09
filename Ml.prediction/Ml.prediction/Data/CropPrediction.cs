using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ml.prediction.Data
{
    public class CropPrediction
    {
        [ColumnName("Score")]
        public float PredictedYield { get; set; }
    }
}
