using System;
using System.Linq;
using System.Threading.Tasks;
using ExcelDna.Integration;
using System.Collections;
using System.Collections.Generic;

namespace SimpleStatistics
{
    public static class MatrixHelper
    {
        [ExcelFunction("Concatenate Ranges")]
        public static object[] ConcatRanges(object[] range1, object[] range2)
        {
            return range1.Concat(range2)
                .Where(v => v != ExcelMissing.Value && v != ExcelEmpty.Value
                    && !typeof(ExcelError).IsAssignableFrom(v.GetType()))
                .ToArray();
        }

        public static int DistinctCount(object[] range1, object[] range2)
        {
            return ConcatRanges(range1, range2).Distinct().Count();
        }

        public static T[] CreateArray<T>(int size, T value)
        {
            var x = new T[size];
            Parallel.For(0, size, i => x[i] = value);
            return x;
        }

        public static object MatrixIndex(string matrix, int row, int column)
        {
            try
            {
                return Double.Parse(matrix.Split('\n').ElementAt(row - 1).Split(';').ElementAt(column - 1));
            }
            catch
            {
                return ExcelError.ExcelErrorNA;
            }
        }

        public static IEnumerable<T> Elements<T>(this T[,,] matrix)
        {
            var n1 = matrix.GetLength(0);
            var n2 = matrix.GetLength(0);
            var n3 = matrix.GetLength(0);
            for (int i = 0; i < n1; i++)
            {
                for (int j = 0; j < n2; j++)
                {
                    for (int k = 0; k < n3; k++)
                    {
                        yield return matrix[i, j, k];
                    }
                }
            }
        }
    }
}
