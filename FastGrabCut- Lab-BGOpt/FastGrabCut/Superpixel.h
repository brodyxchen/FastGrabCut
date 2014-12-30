#pragma once

#include "stdafx.h"


class SLICO
{
public:
	SLICO();
	virtual ~SLICO();


	void DoSuperpixelSegmentation_ForGivenMat( const Mat& img, vector<int>& klabels, int& numlabels);

	void PerformSLICO_ForGivenStepSize(
		const unsigned int*			ubuff,
		const int					width,
		const int					height,
		int*						klabels,
		int&						numlabels,
		const int&					STEP,
		const double&				m);

	void PerformSLICO_ForGivenK(
		const unsigned int*			ubuff,
		const int					width,
		const int					height,
		int*						klabels,
		int&						numlabels,
		const int&					K,
		const double&				m);

	void GetArcAndCenterOfSuperpixels( const Mat& img, vector<int>& klabels, int& numlabels, vector<vector<int> >& arcs, vector<Vec6d>& centers, vector<vector<int> >& spLabels);

	void DrawContoursAroundSegments(Mat& mat, vector<int> klabels, Scalar color);
	void DrawAverageColor(Mat &mat, vector<int>& klabels, vector<Vec6d>& centers);

private:

	void PerformSuperpixelSegmentation_VariableSandM(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*						klabels,
		const int&					STEP,
		const int&					NUMITR);

	void GetLABXYSeeds_ForGivenStepSize(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edgemag);

	void GetLABXYSeeds_ForGivenK(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edges);


	void PerturbSeeds(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		const vector<double>&		edges);

	void DetectLabEdges(
		const double*				lvec,
		const double*				avec,
		const double*				bvec,
		const int&					width,
		const int&					height,
		vector<double>&				edges);

	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);

	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);

	void DoRGBtoLABConversion(
		const unsigned int*&		ubuff,
		double*&					lvec,
		double*&					avec,
		double*&					bvec);



	void EnforceLabelConnectivity(
		const int*					labels,
		const int&					width,
		const int&					height,
		int*						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user


private:
	int										m_width;
	int										m_height;
	int										m_depth;

	int										m_size;

	double*									m_lvec;
	double*									m_avec;
	double*									m_bvec;

	double**								m_lvecvec;
	double**								m_avecvec;
	double**								m_bvecvec;

};

