#pragma once

#include "stdafx.h"

#include "GrabCut.h"
#include "Superpixel.h"



class DoCutConnect
{
private:

	Mat image;
	Mat spImage;
	vector<int> klabels;			//像素的超像素标记					这里是整图的原始标记，后期多使用rect范围内的新标记
	int numlabels;					//超像素总个数

	vector<vector<int> > superarcs;
	vector<Vec6d> superpixels;			//BGRAXY

	vector<vector<int> > spContains;	//每个超像素包含哪些像素

	vector<int> spRelation;			//新旧超像素标记对应关系  spRelation[旧/原始] = spRelation[新]
	int spNum;

	Scalar Superpixels_Contour_Color;
	Scalar Segment_Contour_Color;
	Scalar Rect_Color;
	Scalar Fore_Line_Color;
	Scalar Fore_Pr_Line_Color;
	Scalar Back_Line_Color;
	Scalar Back_Pr_Line_Color;


	Rect rect;
	set<MyPoint> fgdPixels, bgdPixels, prFgdPixels, prBgdPixels;

	vector<int> spMask;		//rect范围的超像素mask
	Mat mask;				//rect范围的像素mask
	Mat fullMask;			//整副图像的像素mask

	GMM fgdGMM;
	GMM bgdGMM;
	SuperpixelGrabCut spGrabcut;
	double beta;

	bool isFirstPxlCut;
	SLICO slico;

	PixelsGrabCut pxlGrabcut;

private:
	


public:

	DoCutConnect();
	void setColor(Scalar* fgColor, Scalar* bgColor, Scalar* ctColor);
	void setImage(Mat& inImage);

	double doSuperpixel();
	Mat doDrawArcAndOther();
	void doSuperpixelSegmentation(Rect* _rect, set<MyPoint>* _fgdPixels, set<MyPoint>* _bgdPixels, set<MyPoint>* _prFgdPixels, set<MyPoint>* _prBgdPixels, int iterCount = 1);
	void doPixelSegmentation();






private:

	int getRealSPLabel(int pixelIndex);

	void initSPMask(Rect* _rect);
	void updateSPMask(set<MyPoint>* _fgdPixels, set<MyPoint>* _bgdPixels, set<MyPoint>* _prFgdPixels, set<MyPoint>* _prBgdPixels);

	void prepareLaterGrabCut(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<int>& spMask);


	void setFullMaskFromSPMask();
	void setFullMaskFromMask();


public:
	Mat getImage(bool isSPImage=true);
	Mat getFullMask();

};

