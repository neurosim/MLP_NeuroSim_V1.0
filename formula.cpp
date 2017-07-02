/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <vector>

/* Activation function */
double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

/* Truncation with a custom threshold */
double truncate(double x, int numLevel, double threshold) {
	if (numLevel <= 0) {   // No truncation if numLevel <= 0
		return x;
	} else {
		int sign = 1;
		if (x<0)
			sign = -1;	// For truncation on negative number

		double val = x * numLevel * sign;
		int r_val = (int)(val);
		if (val - r_val >= threshold)
			val = r_val + 1;
		else
			val = r_val;
		return val*sign/numLevel;
	}
}

/* Round with a custom threshold */
double round_th(double x, double threshold) {
	int sign = 1;
	if (x<0)
		sign = -1;  // For rounding on negative number

	double val = x * sign;
	int r_val = (int)(val);
	if (val - r_val >= threshold)
		val = r_val + 1;
	else
		val = r_val;
	return val * sign;
}

/* Get the conductance in the LTP curve given a pulse position xPulse */
double LTP(double xPulse, int maxNumLevel, double A, double B, double minConductance) {
	return B * (1 - exp(-xPulse/A)) + minConductance;
}

/* Get the conductance in the LTD curve given a pulse position xPulse */
double LTD(double xPulse, int maxNumLevel, double A, double B, double maxConductance) {
	return -B * (1 - exp((xPulse-maxNumLevel)/A)) + maxConductance;
}

/* Inverse LTP: get the pulse position based on the conductance of the LTP curve */
double InvLTP(double conductance, int maxNumLevel, double A, double B, double minConductance) {
	return -A * log(1 - (conductance-minConductance)/B);
}

/* Inverse LTD: get the pulse position based on the conductance of the LTD curve */
double InvLTD(double conductance, int maxNumLevel, double A, double B, double maxConductance) {
	return A * log(1 + (conductance-maxConductance)/B) + maxNumLevel;
}

/* Get the conductance in the LTP data of measured device given a pulse position xPulse */
double MeasuredLTP(double xPulse, int maxNumLevel, std::vector<double>& dataConductanceLTP) {
	if (xPulse > maxNumLevel) {
		xPulse = maxNumLevel;
	} else if (xPulse < 0) {
		xPulse = 0;
	}
	int x1 = (int)xPulse;
	int x2 = (int)ceil(xPulse);
	return (dataConductanceLTP[x2] - dataConductanceLTP[x1]) * (xPulse - x1) + dataConductanceLTP[x1];
}

/* Get the conductance in the LTD data of measured device given a pulse position xPulse */
double MeasuredLTD(double xPulse, int maxNumLevel, std::vector<double>& dataConductanceLTD) {
	if (xPulse > maxNumLevel) {
		xPulse = maxNumLevel;
	} else if (xPulse < 0) {
		xPulse = 0;
	}
	int x1 = (int)xPulse;
	int x2 = (int)ceil(xPulse);
	return (dataConductanceLTD[x2] - dataConductanceLTD[x1]) * (xPulse - x1) + dataConductanceLTD[x1];
}

/* Inverse LTP: get the pulse position based on the LTP conductance data of measured device */
double InvMeasuredLTP(double conductance, int maxNumLevel, std::vector<double>& dataConductanceLTP) {
	int xLeft;	// The nearest integer pulse position on the left
	/* Sweep the data points to find out the pulse position */
	for (int i=0; i<=maxNumLevel-1; i++) {
		if ((dataConductanceLTP[i] - conductance) * (dataConductanceLTP[i+1] - conductance) <= 0) {
			xLeft = i;
			break;
		}
	}
	return xLeft + (conductance - dataConductanceLTP[xLeft])/(dataConductanceLTP[xLeft+1] - dataConductanceLTP[xLeft]);
}

/* Inverse LTD: get the pulse position based on the LTD conductance data of measured device */
double InvMeasuredLTD(double conductance, int maxNumLevel, std::vector<double>& dataConductanceLTD) {
	int xLeft;  // The nearest integer pulse position on the left
	/* Sweep the data points to find out the pulse position */
	for (int i=0; i<=maxNumLevel-1; i++) {
		if ((dataConductanceLTD[i] - conductance) * (dataConductanceLTD[i+1] - conductance) <= 0) {
			xLeft = i;
			break;
		}
	}
	return xLeft + (conductance - dataConductanceLTD[xLeft])/(dataConductanceLTD[xLeft+1] - dataConductanceLTD[xLeft]);
}

/* Weight update nonlinearity baselines */
double getParamA(double NL) {
	int index = (int)(NL * 100) - 1;	// -1 because index starts from 0
	/* This paramA table corresponds to nonlinearity label from 0.01 to 7, with step=0.01 */
	double data[] = {7999.0165,	3999.4976,	2666.3199,	1999.7275,	1599.7692,	1333.1279,	1142.6679,	999.8211,	888.7164,	799.8312,
					727.1057,	666.5000,	615.2170,	571.2593,	533.1616,	499.8252,	470.4099,	444.2622,	420.8661,	399.8089,
					380.7565,	363.4355,	347.6200,	333.1219,	319.7831,	307.4698,	296.0681,	285.4802,	275.6221,	266.4207,
					257.8125,	249.7418,	242.1599,	235.0235,	228.2945,	221.9390,	215.9266,	210.2302,	204.8257,	199.6909,
					194.8064,	190.1541,	185.7178,	181.4828,	177.4358,	173.5644,	169.8575,	166.3047,	162.8966,	159.6246,
					156.4806,	153.4572,	150.5477,	147.7456,	145.0452,	142.4410,	139.9279,	137.5012,	135.1565,	132.8898,
					130.6971,	128.5750,	126.5199,	124.5289,	122.5989,	120.7272,	118.9111,	117.1483,	115.4363,	113.7731,
					112.1564,	110.5845,	109.0555,	107.5676,	106.1192,	104.7087,	103.3346,	101.9956,	100.6904,	99.4175,
					98.1760,	96.9645,	95.7820,	94.6276,	93.5001,	92.3987,	91.3224,	90.2704,	89.2419,	88.2361,
					87.2523,	86.2896,	85.3476,	84.4254,	83.5224,	82.6382,	81.7720,	80.9233,	80.0917,	79.2765,
					78.4773,	77.6937,	76.9251,	76.1712,	75.4315,	74.7056,	73.9931,	73.2937,	72.6070,	71.9327,
					71.2704,	70.6197,	69.9805,	69.3523,	68.7350,	68.1281,	67.5315,	66.9449,	66.3681,	65.8007,
					65.2426,	64.6935,	64.1532,	63.6215,	63.0982,	62.5831,	62.0760,	61.5767,	61.0850,	60.6008,
					60.1239,	59.6540,	59.1911,	58.7351,	58.2856,	57.8427,	57.4061,	56.9758,	56.5515,	56.1332,
					55.7207,	55.3140,	54.9128,	54.5170,	54.1267,	53.7415,	53.3616,	52.9866,	52.6166,	52.2514,
					51.8910,	51.5352,	51.1839,	50.8371,	50.4947,	50.1566,	49.8227,	49.4930,	49.1673,	48.8455,
					48.5277,	48.2137,	47.9034,	47.5969,	47.2940,	46.9946,	46.6987,	46.4063,	46.1172,	45.8315,
					45.5490,	45.2697,	44.9935,	44.7204,	44.4504,	44.1833,	43.9192,	43.6580,	43.3996,	43.1439,
					42.8910,	42.6409,	42.3933,	42.1484,	41.9060,	41.6662,	41.4288,	41.1939,	40.9614,	40.7313,
					40.5034,	40.2779,	40.0547,	39.8336,	39.6148,	39.3981,	39.1835,	38.9710,	38.7606,	38.5522,
					38.3458,	38.1413,	37.9388,	37.7382,	37.5395,	37.3427,	37.1476,	36.9544,	36.7630,	36.5733,
					36.3853,	36.1990,	36.0144,	35.8314,	35.6501,	35.4704,	35.2922,	35.1157,	34.9406,	34.7671,
					34.5951,	34.4246,	34.2555,	34.0879,	33.9217,	33.7568,	33.5934,	33.4313,	33.2706,	33.1112,
					32.9531,	32.7963,	32.6408,	32.4866,	32.3336,	32.1818,	32.0312,	31.8819,	31.7337,	31.5867,
					31.4408,	31.2961,	31.1525,	31.0100,	30.8686,	30.7283,	30.5891,	30.4509,	30.3137,	30.1776,
					30.0426,	29.9085,	29.7754,	29.6433,	29.5122,	29.3820,	29.2528,	29.1245,	28.9972,	28.8708,
					28.7452,	28.6206,	28.4969,	28.3740,	28.2520,	28.1309,	28.0105,	27.8911,	27.7724,	27.6546,
					27.5376,	27.4214,	27.3060,	27.1913,	27.0775,	26.9644,	26.8520,	26.7404,	26.6296,	26.5194,
					26.4100,	26.3013,	26.1934,	26.0861,	25.9795,	25.8736,	25.7684,	25.6638,	25.5600,	25.4567,
					25.3542,	25.2522,	25.1509,	25.0503,	24.9503,	24.8508,	24.7520,	24.6538,	24.5562,	24.4592,
					24.3628,	24.2670,	24.1717,	24.0770,	23.9829,	23.8893,	23.7963,	23.7039,	23.6119,	23.5206,
					23.4297,	23.3394,	23.2496,	23.1603,	23.0715,	22.9832,	22.8955,	22.8082,	22.7214,	22.6351,
					22.5493,	22.4640,	22.3791,	22.2947,	22.2108,	22.1273,	22.0443,	21.9618,	21.8797,	21.7980,
					21.7168,	21.6360,	21.5556,	21.4757,	21.3962,	21.3171,	21.2384,	21.1602,	21.0823,	21.0049,
					20.9278,	20.8512,	20.7749,	20.6991,	20.6236,	20.5485,	20.4738,	20.3995,	20.3255,	20.2519,
					20.1787,	20.1058,	20.0333,	19.9612,	19.8894,	19.8180,	19.7469,	19.6762,	19.6058,	19.5357,
					19.4660,	19.3966,	19.3275,	19.2588,	19.1904,	19.1223,	19.0546,	18.9871,	18.9200,	18.8532,
					18.7867,	18.7205,	18.6546,	18.5890,	18.5237,	18.4587,	18.3940,	18.3296,	18.2655,	18.2017,
					18.1381,	18.0748,	18.0119,	17.9492,	17.8867,	17.8246,	17.7627,	17.7011,	17.6397,	17.5786,
					17.5178,	17.4573,	17.3970,	17.3369,	17.2771,	17.2176,	17.1583,	17.0993,	17.0405,	16.9820,
					16.9237,	16.8656,	16.8078,	16.7502,	16.6928,	16.6357,	16.5788,	16.5222,	16.4658,	16.4096,
					16.3536,	16.2979,	16.2423,	16.1870,	16.1320,	16.0771,	16.0224,	15.9680,	15.9138,	15.8597,
					15.8059,	15.7523,	15.6989,	15.6457,	15.5927,	15.5400,	15.4874,	15.4350,	15.3828,	15.3308,
					15.2790,	15.2274,	15.1759,	15.1247,	15.0737,	15.0228,	14.9721,	14.9216,	14.8713,	14.8212,
					14.7713,	14.7215,	14.6719,	14.6225,	14.5733,	14.5242,	14.4753,	14.4266,	14.3781,	14.3297,
					14.2815,	14.2335,	14.1856,	14.1379,	14.0903,	14.0429,	13.9957,	13.9487,	13.9018,	13.8550,
					13.8084,	13.7620,	13.7157,	13.6696,	13.6236,	13.5778,	13.5321,	13.4866,	13.4412,	13.3960,
					13.3509,	13.3060,	13.2612,	13.2165,	13.1720,	13.1277,	13.0834,	13.0394,	12.9954,	12.9516,
					12.9079,	12.8644,	12.8210,	12.7777,	12.7346,	12.6916,	12.6487,	12.6060,	12.5634,	12.5209,
					12.4786,	12.4363,	12.3942,	12.3523,	12.3104,	12.2687,	12.2271,	12.1856,	12.1443,	12.1030,
					12.0619,	12.0209,	11.9801,	11.9393,	11.8987,	11.8582,	11.8177,	11.7775,	11.7373,	11.6972,
					11.6573,	11.6174,	11.5777,	11.5381,	11.4986,	11.4592,	11.4199,	11.3807,	11.3416,	11.3027,
					11.2638,	11.2250,	11.1864,	11.1479,	11.1094,	11.0711,	11.0328,	10.9947,	10.9567,	10.9187,
					10.8809,	10.8432,	10.8055,	10.7680,	10.7306,	10.6932,	10.6560,	10.6188,	10.5818,	10.5448,
					10.5079,	10.4712,	10.4345,	10.3979,	10.3614,	10.3250,	10.2887,	10.2525,	10.2163,	10.1803,
					10.1444,	10.1085,	10.0727,	10.0370,	10.0014,	9.9659,		9.9305,		9.8951,		9.8599,		9.8247,
					9.7896,		9.7546,		9.7197,		9.6849,		9.6501,		9.6154,		9.5809,		9.5463,		9.5119,		9.4776,
					9.4433,		9.4091,		9.3750,		9.3410,		9.3070,		9.2731,		9.2393,		9.2056,		9.1720,		9.1384,
					9.1049,		9.0715,		9.0382,		9.0049,		8.9717,		8.9386,		8.9055,		8.8726,		8.8397,		8.8069,
					8.7741,		8.7414,		8.7088,		8.6763,		8.6438,		8.6114,		8.5791,		8.5468,		8.5147,		8.4825,
					8.4505,		8.4185,		8.3866,		8.3548,		8.3230,		8.2913,		8.2597,		8.2281,		8.1966,		8.1652,
					8.1338,		8.1025,		8.0712,		8.0401,		8.0090,		7.9779,		7.9469,		7.9160,		7.8852,		7.8544,
					7.8237,		7.7930,		7.7624,		7.7319,		7.7014,		7.6710,		7.6406,		7.6104,		7.5801,		7.5500,
					7.5199,		7.4898,		7.4599,		7.4299,		7.4001,		7.3703,		7.3406,		7.3109,		7.2813,		7.2517,
					7.2222,		7.1928,		7.1634,		7.1341,		7.1048,		7.0756,		7.0464,		7.0173,		6.9883,		6.9593,
					6.9304,		6.9016,		6.8728,		6.8440,		6.8153,		6.7867,		6.7581,		6.7296,		6.7011,		6.6727,
					6.6444,		6.6161,		6.5878,		6.5596,		6.5315,		6.5034,		6.4754,		6.4474,		6.4195,		6.3917,
					6.3639,		6.3361,		6.3084,		6.2808,		6.2532,		6.2257,		6.1982,		6.1707,		6.1434,		6.1161,
					6.0888,		6.0616,		6.0344,		6.0073,		5.9802,		5.9532,		5.9263,		5.8994,		5.8726,		5.8458,
					5.8190,		5.7923,		5.7657,		5.7391,		5.7126,		5.6861,		5.6597,		5.6333,		5.6070,		5.5807};
	if (index < 0) {
		index = 0;
	} else if (index > 699) {
		index = 699;
	}
	return data[index];
}

double NonlinearConductance(double C, double NL, double Vw, double Vr, double V) {   // Nonlinearity is the current ratio between Vw and V, and C means the resistance at Vr
	double C_NL = C * Vr/V * pow(NL, (V-Vr)/(Vw/2));
	return C_NL;
}
