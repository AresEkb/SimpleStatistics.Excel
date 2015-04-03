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
using System.Text;
using MathNet.Numerics.LinearAlgebra.Double;

namespace SimpleStatistics
{
    class Program
    {
        static void Main(string[] args)
        {
            //var d = new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
            //var p = new double[,] { { 47.7 }, { 43.0 }, { 15.0 }, { 47.5 }, { 45.8 }, { 10.2 }, { 30.0 }, { 33.0 }, { 35.1 }, { 33.8 }, { 25.2 }, { 48.0 }, { 69.7 }, { 15.0 }, { 28.0 }, { 52.0 }, { 45.7 }, { 70.9 }, { 25.0 }, { 12.8 }, { 17.1 }, { 25.8 }, { 36.0 }, { 33.2 }, { 76.0 }, { 49.1 }, { 19.6 }, { 29.4 }, { 66.0 }, { 72.0 }, { 47.0 }, { 86.0 }, { 23.4 }, { 50.0 }, { 50.0 }, { 24.6 }, { 95.0 }, { 30.6 }, { 90.1 }, { 35.4 }, { 92.0 }, { 25.8 }, { 53.4 }, { 113.0 }, { 33.6 }, { 28.8 }, { 113.0 }, { 18.6 }, { 72.0 }, { 77.4 }, { 75.0 }, { 55.5 }, { 54.7 }, { 54.0 }, { 37.8 }, { 47.6 }, { 53.6 }, { 85.9 }, { 86.0 }, { 63.5 }, { 154.7 }, { 40.8 }, { 29.4 }, { 41.4 }, { 60.0 }, { 27.0 }, { 27.0 }, { 109.0 }, { 46.0 }, { 80.8 }, { 68.8 }, { 163.0 } };
            //var b = LogisticRegression.EstimateLogisticRegression(d, p);
            //if (b is double[])
            //{
            //    Console.WriteLine(new DenseVector((double[])b));
            //}
            //else
            //{
            //    Console.WriteLine(b);
            //}

            //var controls = new object[] { 47.7, 43.0, 15.0, 47.5, 45.8, 10.2, 30.0, 33.0, 35.1, 33.8, 25.2, 48.0, 69.7, 15.0, 28.0, 52.0, 45.7, 70.9, 25.0, 12.8, 17.1, 25.8, 36.0, 33.2, 76.0, 49.1, 19.6, 29.4, 66.0, 72.0, 47.0 };
            //var cases = new object[] { 86.0, 23.4, 50.0, 50.0, 24.6, 95.0, 30.6, 90.1, 35.4, 92.0, 25.8, 53.4, 113.0, 33.6, 28.8, 113.0, 18.6, 72.0, 77.4, 75.0, 55.5, 54.7, 54.0, 37.8, 47.6, 53.6, 85.9, 86.0, 63.5, 154.7, 40.8, 29.4, 41.4, 60.0, 27.0, 27.0, 109.0, 46.0, 80.8, 68.8, 163.0 };
            //var q = LogisticRegression.EstimateLogisticRegression2(controls, cases);
            //Console.WriteLine(q);
            //if (q is double[])
            //{
            //    Console.WriteLine(new DenseVector((double[])q));
            //}
            //else
            //{
            //    Console.WriteLine(q);
            //}

            Console.ReadKey();
        }
    }
}
