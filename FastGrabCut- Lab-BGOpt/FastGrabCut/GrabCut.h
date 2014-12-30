#pragma once

#include "stdafx.h"

#include "GrabCut_Helper.h"

class SuperpixelGrabCut
{
private:
	Vec3d Vec6d_to_Vec3d(Vec6d v6);
	Vec2d Vec6d_to_Vec2d(Vec6d v6);
	void calcNWeights(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<vector<double> >& spNWeights, int& sideNum, double beta, double gamma);
		 
	void checkMask(const vector<Vec6d>& superpixels, const vector<int>& spMask );
	void initMaskWithRect(vector<int>& klabels,  int width, vector<int>& mask, int size, Rect rect );

	void initGMMs(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM );

	void assignGMMsComponents(vector<Vec6d>& superpixels, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM, vector<int>& compIdxs);

	void learnGMMs(vector<Vec6d>& superpixels, vector<int>& spMask, vector<int>& compIdxs,  GMM& bgdGMM, GMM& fgdGMM );


	void constructGCGraph(vector<Vec6d>& superpixels, vector<int>& spMask, GMM& bgdGMM, GMM& fgdGMM, double lambda, vector<vector<int> >& superarcs, vector<vector<double> >& spNWeights, vector<int>& vtxs, int sideNum, GCGraph<double>& graph);

	void estimateSegmentation( GCGraph<double>& graph, vector<int>& spMask, vector<int> vtxs);

public:
	double calcBeta( const Mat& img );
	void grabCut_ForSuperpixels(vector<int>& klabels, vector<Vec6d> superpixels, vector<vector<int> >& superarcs, double beta, vector<int>& spMask, GMM& fgdGMM, GMM& bgdGMM, int iterCount, int mode=GC_EVAL);
};


////////////////////////////////
class PixelsGrabCut
{
private:
	void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma );
	void checkMask( const Mat& img, const Mat& mask );
	void initMaskWithRect( Mat& mask, Size imgSize, Rect rect );
    void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs );
	void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM );
	void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, GCGraph<double>& graph );
	void estimateSegmentation( GCGraph<double>& graph, Mat& mask );
	double calcBeta( const Mat& img );
public:
	void grabCut_ForPixel( Mat& img, Mat& mask, Rect& rect, GMM& fgdGMM, GMM& bgdGMM,  int iterCount, bool isFirst=false);
};