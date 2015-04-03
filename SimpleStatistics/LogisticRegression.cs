/**
 * Copyright (c) 2011, 2015 Denis Nikiforov.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *    Denis Nikiforov - initial API and implementation
 */

using System;
using System.Collections.Generic;
using System.Linq;
using ExcelDna.Integration;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SimpleStatistics
{
    public enum ScaleType { Unknown = 0, Binary, Nominal, Ordinal, Interval, Ratio }

    public static class SimpleStatistics
    {
        private static IEnumerable<double> FractionalRanking<T>(this IEnumerable<T> seq)
            where T : IComparable
        {
            return seq.Select(x => seq.Count(y => y.CompareTo(x) < 0)
                + ((double)seq.Count(y => y.Equals(x))) / 2 + 0.5);
        }

        private static IEnumerable<double> FractionalRankingOld<T>(this IEnumerable<T> seq)
            where T : IComparable
        {
            return seq.Select(x => ((double)seq.Count(y => y.CompareTo(x) < 0))
                + seq.Count(y => y.Equals(x)) / 2);
        }

        private static IEnumerable<double> SubtractSamples(object[] minuend, object[] subtrahend)
        {
            var n = Math.Min(minuend.Count(), subtrahend.Count());
            for (var i = 0; i < n; i++)
            {
                if (minuend[i] is double && subtrahend[i] is double)
                {
                    yield return (double)minuend[i] - (double)subtrahend[i];
                }
            }
        }

        public static object SignTest(object[] before, object[] after)
        {
            var len = Math.Min(before.Count(), after.Count());
            var pos = 0;
            var neg = 0;
            for (var i = 0; i < len; i++)
            {
                if (before[i] is double && after[i] is double)
                {
                    var b = (double)before[i];
                    var a = (double)after[i];
                    if (b < a) pos++;
                    else if (b > a) neg++;
                }
            }
            var x = Math.Min(pos, neg);
            var n = pos + neg;
            if (n < 25)
            {
                //var p = Math.Min(new Binomial(0.5, n).CumulativeDistribution(x) * 2, 1);
                //return new DenseVector(new double[] { Double.NaN, p }).ToString();
                return Math.Min(new Binomial(0.5, n).CumulativeDistribution(x) * 2, 1);
            }
            else
            {
                // http://blog.excelmasterseries.com/2010/09/sign-test-nonparametric-in-excel.html
                //var z = (x + 0.5 - n + 2) * 2 / Math.Sqrt(n);

                // http://www.mathworks.com/help/stats/signtest.html
                //var z = (2 * x - n - Math.Sign(pos - neg)) / Math.Sqrt(n);

                // This one is seems to be used in SPSS
                // http://www.fon.hum.uva.nl/Service/Statistics/Sign_Test.html
                var z = (1 - Math.Abs(pos - neg)) / Math.Sqrt(n);
                //var p = (1 - new Normal().CumulativeDistribution(Math.Abs(z))) * 2;
                //return new DenseVector(new double[] { z, p }).ToString();
                return (1 - new Normal().CumulativeDistribution(Math.Abs(z))) * 2;
            }
        }

        public static object WilcoxonSignedRankTest(object[] before, object[] after)
        {
            var diff = SubtractSamples(after, before).Where(d => d != 0).ToList();
            var n = diff.Count;
            var sign = new DenseVector(diff.Select(d => (double)Math.Sign(d)).ToArray());
            var rank = new DenseVector(diff.Select(d => Math.Abs(d)).FractionalRanking().ToArray());
            var w = sign * rank;
            if (n < 10)
            {
                switch (n)
                {
                    case 6:
                        if (w <= 21) return 0.05;
                        break;
                    case 7:
                        if (w <= 24) return 0.05;
                        else if (w <= 28) return 0.02;
                        break;
                    case 8:
                        if (w <= 30) return 0.05;
                        else if (w <= 34) return 0.02;
                        else if (w <= 36) return 0.01;
                        break;
                    case 9:
                        if (w <= 35) return 0.05;
                        else if (w <= 39) return 0.02;
                        else if (w <= 43) return 0.01;
                        break;
                }
                return ExcelError.ExcelErrorNA;
            }
            else
            {
                // Its seems that SPSS doesn't use correction for continuity
                //var z = w / Math.Sqrt(n * (n + 1) * (2 * n + 1) / 6);
                // http://vassarstats.net/textbook/ch12a.html
                var z = (w - Math.Sign(w) * 0.5) / Math.Sqrt(n * (n + 1) * (2 * n + 1) / 6);
                //var z = (w - 0.5 - n * (n + 1) / 4) / Math.Sqrt(n * (n + 1) * (2 * n + 1) / 24);
                return (1 - new Normal().CumulativeDistribution(Math.Abs(z))) * 2;
                //return new DenseVector(new double[] { z, (1 - new Normal().CumulativeDistribution(Math.Abs(z))) * 2 }).ToString();
            }
        }

        public static double Atanh(double d)
        {
            return Math.Log((1 + d) / (1 - d)) / 2;
        }

        public static object FourPointCorrelation(int a, int b, int c, int d)
        {
            return (a * d - b * c) / Math.Sqrt((a + b) * (a + c) * (b + d) * (c + d));
        }

        public static object SpearmanRankCorrelation(object[] cases, object[] controls)
        {
            var cases2 = cases.OfType<double>().ToArray();
            var controls2 = controls.OfType<double>().ToArray();
            var n1 = cases2.Count();
            var n2 = controls2.Count();
            var n = n1 + n2;
            var x = MatrixHelper.CreateArray(n1, 1).Concat(MatrixHelper.CreateArray(n2, 0)).ToArray();
            var y = cases2.Concat(controls2).FractionalRankingOld().ToArray();
            var mx = x.Average();
            var my = y.Average();
            double sp = 0, sx = 0, sy = 0;
            for (int i = 0; i < n; i++)
            {
                var dx = x[i] - mx;
                var dy = y[i] - my;
                sp += dx * dy;
                sx += dx * dx;
                sy += dy * dy;
            }
            var rho = sp / Math.Sqrt(sx * sy);
            var fRho = Atanh(rho);
            var z = Math.Sqrt((n - 3) / 1.06) * fRho;
            var p = (1 - new Normal().CumulativeDistribution(Math.Abs(z))) * 2;
            var delta = 1.96 / Math.Sqrt(n - 3);
            //var t = rho * Math.Sqrt((n - 2) / (1 - rho * rho));
            //var p2 = (1 - new StudentT(0, 1, n - 2).CumulativeDistribution(Math.Abs(z))) * 2;
            return new DenseVector(new double[] { rho, p,
                Math.Tanh(fRho - delta), Math.Tanh(fRho + delta) }).ToString();
        }

        public static object FisherExactTest(int a, int b, int c, int d)
        {
            var hg = new Hypergeometric(a + b + c + d, a + b, a + c);
            var p = 0.0;
            if (a * d >= b * c)
            {
                var max = Math.Min(a + b, a + c);
                for (; a <= max; a += 1) p += hg.Probability(a);
            }
            else
            {
                for (; a >= 0; a -= 1) p += hg.Probability(a);
            }
            return p;
        }

        //        Function MannWhitneyU(sample1 As Range, sample2 As Range) As Double
        //    Dim n1 As Long
        //    Dim n2 As Long
        //    Dim u1 As Double
        //    Dim v1 As Variant
        //    n1 = WorksheetFunction.Count(sample1)
        //    n2 = WorksheetFunction.Count(sample2)
        //    For Each v1 In sample1
        //        If WorksheetFunction.IsNumber(v1) Then
        //            u1 = u1 + WorksheetFunction.CountIf(sample2, "<" & Str(v1)) _
        //                    + WorksheetFunction.CountIf(sample2, "=" & Str(v1)) / 2
        //        End If
        //    Next v1
        //    MannWhitneyU = (u1 - n1 * n2 / 2) / Sqr(n1 * n2 * (n1 + n2 + 1) / 12)
        //End Function
        //Function MannWhitneyU2(sample1 As Range, sample2 As Range) As Double
        //    Dim n1 As Long
        //    Dim n2 As Long
        //    Dim sample As Range
        //    Dim r1 As Double
        //    Dim v1 As Variant
        //    n1 = WorksheetFunction.Count(sample1)
        //    n2 = WorksheetFunction.Count(sample2)
        //    Set sample = Union(sample1, sample2)
        //    For Each v1 In sample1
        //        If WorksheetFunction.IsNumber(v1) Then
        //            r1 = r1 + WorksheetFunction.CountIf(sample, "<" & Str(v1)) _
        //                    + WorksheetFunction.CountIf(sample, "=" & Str(v1)) / 2 _
        //                    + 1 / 2
        //        End If
        //    Next v1
        //    MannWhitneyU2 = (r1 - n1 * (n1 + 1) / 2 - n1 * n2 / 2) / Sqr(n1 * n2 * (n1 + n2 + 1) / 12)
        //End Function


        //public static object KolmogorovSmirnovTest(double[] samples)
        //{
        //    if (!samples.Any()) return false;

        //    var vals = samples.ToList();
        //    Accumulator acc = new Accumulator(vals.ToArray());
        //    double dmax = double.MinValue;
        //    double cv = 0;

        //    var test = new MathNet.Numerics.Distributions.Normal(acc.Mean, acc.Sigma);

        //    // the 0 entry is to force the list to be a base 1 index table.
        //    var cvTable = new List<double>() { 0, .975, .842, .708, .624, .565,
        //                                        .521, .486, .457, .432, .410,
        //                                        .391, .375, .361, .349, .338,
        //                                        .328, .318, .309, .301, .294};

        //    test.EstimateDistributionParameters(DataSet.Values.ToArray());
        //    vals.Sort();

        //    for (int i = 0; i < vals.Count; i++)
        //    {
        //        double dr = Math.Abs(((i + 1) / (double)vals.Count) - test.CumulativeDistribution(vals[i]));
        //        double dl = Math.Abs(test.CumulativeDistribution(vals[i]) - (i / (double)vals.Count));
        //        dmax = Math.Max(dmax, Math.Max(dl, dr));
        //    }

        //    // get critical value and compare to d(N)
        //    if (vals.Count <= 10)
        //        cv = cvTable[vals.Count];
        //    else if (vals.Count > 10)
        //        cv = 1.36 / Math.Sqrt(vals.Count);

        //    return (dmax < cv);
        //}

        //public static object ShapiroWilkTest(double[] samples)
        //{
        //    var x = new DenseVector(samples);
        //    var m = Statistics.Mean(samples);
        //    var denum = x.Select(xi => (xi - m) * (xi - m)).Sum();

        //    x.Select((xi, i) => Statistics.Mean(Statistics.OrderStatistic(Normal.Samples(new Random(), 0, 1), i + 1))
        //        .Select(s => s, i+1)).Take(50));
        //    for (int )
        //    var a 
        //    var xith = a.Select((ai, i) => ai * Statistics.OrderStatistic(x, i)).Sum();
        //}

        public static object LogisticRegressionCC(object[] cases, object[] controls, int scaleType)
        {
            double[] cases2;
            double[] controls2;
            if (scaleType == (int)ScaleType.Binary)
            {
                cases2 = cases.OfType<double>().Select(x => x == 0 ? 0.0 : 1.0).ToArray();
                controls2 = controls.OfType<double>().Select(x => x == 0 ? 0.0 : 1.0).ToArray();
                if (!cases2.Any(x => x == 0) || !cases2.Any(x => x == 1) ||
                    !controls2.Any(x => x == 0) || !controls2.Any(x => x == 1))
                {
                    return ExcelError.ExcelErrorDiv0;
                }
            }
            else
            {
                cases2 = cases.OfType<double>().ToArray();
                controls2 = controls.OfType<double>().ToArray();
            }
            var n1 = cases2.Count();
            var n2 = controls2.Count();
            if (n1 == 0 || n2 == 0) return ExcelError.ExcelErrorNA;
            return LogisticRegression(
                new DenseVector(n1, 1).Concat(new DenseVector(n2, 0)).ToArray(),
                new DenseMatrix(n1, 1, cases2.ToArray())
                    .Stack(new DenseMatrix(n2, 1, controls2.ToArray())).ToArray());
        }

        public static IEnumerable<Tuple<double, double>> ZipVariables(this IEnumerable<object> var1, IEnumerable<object> var2)
        {
            return var1.Zip(var2, (x, y) => new Tuple<object, object>(x, y))
                .Where(t => t.Item1 is double && t.Item2 is double)
                .Select(t => new Tuple<double, double>((double)t.Item1, (double)t.Item2));
        }

        public static IEnumerable<Tuple<double, double, double>> ZipVariables(this IEnumerable<object> var1, IEnumerable<object> var2, IEnumerable<object> var3)
        {
            return var1.Zip(var2, (x, y) => new Tuple<object, object>(x, y))
                .Zip(var3, (t, z) => new Tuple<object, object, object>(t.Item1, t.Item2, z))
                .Where(t => t.Item1 is double && t.Item2 is double && t.Item3 is double)
                .Select(t => new Tuple<double, double, double>((double)t.Item1, (double)t.Item2, (double)t.Item3));
        }

        public static object LogisticRegression2(object[] response, object[] predictor)
        {
            var vars = response.ZipVariables(predictor);
            var n = vars.Count();
            return LogisticRegression(vars.Select(t => t.Item1).ToArray(),
                new DenseMatrix(n, 1, vars.Select(t => t.Item2).ToArray()).ToArray());
        }

        public static object LogisticRegression3(object[] response, object[] predictor, object[] covariate)
        {
            var vars = response.ZipVariables(predictor, covariate);
            var n = vars.Count();
            var pred = new DenseMatrix(n, 2);
            pred.SetColumn(0, vars.Select(t => t.Item2).ToArray());
            pred.SetColumn(1, vars.Select(t => t.Item3).ToArray());
            return LogisticRegression(vars.Select(t => t.Item1).ToArray(), pred.ToArray());
        }

        // http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/logit.pdf
        // http://msdn.microsoft.com/en-us/magazine/jj618304.aspx
        [ExcelFunction(Description = "Estimate Logistic Regression", Category = "Simple Statistics")]
        public static object LogisticRegression(double[] dependent, double[,] predictor)
        {
            var n = dependent.Length;
            var predictor2 = new DenseMatrix(predictor);
            //var x = new DenseMatrix(n, 1, 1).Append(
            //    n != predictor.GetLength(0) && n == predictor.GetLength(1)
            //        ? predictor2.Transpose()
            //        : predictor2);
            var x = new DenseMatrix(n, 1, 1).Append(predictor2);
            var dim = x.ColumnCount;
            var y = new DenseVector(dependent);
            var w = new DiagonalMatrix(n);
            var b = new DenseVector(dim);
            var p = new DenseVector(n);
            try
            {
                for (int k = 0; k < 100; k++)
                {
                    for (int i = 0; i < n; i++) p[i] = 1 / (1 + Math.Exp(-b * x.Row(i)));
                    w.SetDiagonal(p.PointwiseMultiply(-p.Subtract(1)));
                    //var xtw = x.TransposeThisAndMultiply(w);
                    //var cov = (xtw * x).Inverse();
                    //var b2 = (DenseVector)(cov * xtw * (x * b + w.Inverse() * (y - p)));
                    var cov = (x.TransposeThisAndMultiply(w) * x).Inverse();
                    var b2 = (DenseVector)(b + cov * x.Transpose() * (y - p));
                    var precision = (b2 - b).AbsoluteMaximum();
                    b = b2;
                    if (precision <= 0.0001)
                    {
                        var se = new DenseVector(cov.Diagonal().Select(covi => Math.Sqrt(covi)).ToArray());
                        var wald = new DenseVector(b.PointwiseMultiply(b).PointwiseDivide(se.PointwiseMultiply(se)));
                        var waldp = new DenseVector(wald.Select(waldi => 1 - new ChiSquare(1).CumulativeDistribution(waldi)).ToArray());
                        var or = new DenseVector(b.Select(bi => Math.Exp(bi)).ToArray());
                        var ci = new DenseMatrix(2, dim);
                        ci.SetRow(0, (b - 1.96 * se).Select(bi => Math.Exp(bi)).ToArray());
                        ci.SetRow(1, (b + 1.96 * se).Select(bi => Math.Exp(bi)).ToArray());
                        return b.ToRowMatrix()
                            .Stack(se.ToRowMatrix())
                            .Stack(wald.ToRowMatrix())
                            .Stack(waldp.ToRowMatrix())
                            .Stack(or.ToRowMatrix())
                            .Stack(ci)
                            .ToString();
                    }
                }
                return ExcelError.ExcelErrorNA;
            }
            catch
            {
                return ExcelError.ExcelErrorValue;
            }
        }

        public static object CochranMantelHaenszelTest(object[] response, object[] predictor, object[] covariate)
        {
            var size = Math.Min(predictor.Length, Math.Min(covariate.Length, response.Length));
            var n = new double[2, 2, 2];
            for (int i = 0; i < size; i++)
            {
                if (predictor[i] is double && covariate[i] is double && response[i] is double)
                {
                    n[(double)predictor[i] == 0 ? 0 : 1, (double)response[i] == 0 ? 0 : 1, (double)covariate[i] == 0 ? 0 : 1]++;
                }
            }
            var corr = n.Elements().Any(x => x == 0) ? 0.5 : 0.0;
            var na = new double[2, 2, 2];
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 2; j++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        na[i, j, k] = n[i, j, k] + corr;
                    }
                }
            }
            var ne = new double[2, 2, 2];
            var chi = new double[2];
            var p = new double[2];
            var or = new double[2];
            var se2 = new double[2];
            var or1 = new double[2];
            var or2 = new double[2];
            var w = new double[2];
            var mu = new double[2];
            var v = new double[2];
            for (int i = 0; i < 2; i++)
            {
                var sum = n[0, 0, i] + n[0, 1, i] + n[1, 0, i] + n[1, 1, i];
                ne[0, 0, i] = (n[0, 0, i] + n[0, 1, i]) * (n[0, 0, i] + n[1, 0, i]) / sum;
                ne[0, 1, i] = (n[0, 0, i] + n[0, 1, i]) * (n[0, 1, i] + n[1, 1, i]) / sum;
                ne[1, 0, i] = (n[1, 0, i] + n[1, 1, i]) * (n[0, 0, i] + n[1, 0, i]) / sum;
                ne[1, 1, i] = (n[1, 0, i] + n[1, 1, i]) * (n[0, 1, i] + n[1, 1, i]) / sum;
                chi[i] = (n[0, 0, i] - ne[0, 0, i]) * (n[0, 0, i] - ne[0, 0, i]) / ne[0, 0, i]
                       + (n[0, 1, i] - ne[0, 1, i]) * (n[0, 1, i] - ne[0, 1, i]) / ne[0, 1, i]
                       + (n[1, 0, i] - ne[1, 0, i]) * (n[1, 0, i] - ne[1, 0, i]) / ne[1, 0, i]
                       + (n[1, 1, i] - ne[1, 1, i]) * (n[1, 1, i] - ne[1, 1, i]) / ne[1, 1, i];
                p[i] = 1 - new ChiSquare(1).CumulativeDistribution(chi[i]);
                or[i] = (na[0, 0, i] * na[1, 1, i]) / (na[0, 1, i] * na[1, 0, i]);
                se2[i] = 1 / na[0, 0, i] + 1 / na[0, 1, i] + 1 / na[1, 0, i] + 1 / na[1, 1, i];
                or1[i] = Math.Exp(Math.Log(or[i]) - 1.96 * Math.Sqrt(se2[i]));
                or2[i] = Math.Exp(Math.Log(or[i]) + 1.96 * Math.Sqrt(se2[i]));
                w[i] = 1 / se2[i];
                mu[i] = n[0, 0, i] - (n[0, 0, i] + n[0, 1, i]) * (n[0, 0, i] + n[1, 0, i]) / sum;
                v[i] = (n[0, 0, i] + n[0, 1, i]) * (n[0, 0, i] + n[1, 0, i]) * (n[0, 1, i] + n[1, 1, i]) * (n[1, 0, i] + n[1, 1, i]) / (sum * (sum * sum - 1));
            }
            var chi0 = Math.Pow(Math.Abs(mu.Sum()) - 0.5, 2) / v.Sum();
            var p0 = 1 - new ChiSquare(1).CumulativeDistribution(chi0);
            var or0 = Math.Exp(w.Zip(or, (x, y) => x * Math.Log(y)).Sum() / w.Sum());
            var or01 = or0 * Math.Exp(-1.96 / Math.Sqrt(w.Sum()));
            var or02 = or0 * Math.Exp(1.96 / Math.Sqrt(w.Sum()));

            return new DenseVector(new double[] { chi0, p0, or0, or01, or02 }).ToString();
        }
    }
}
