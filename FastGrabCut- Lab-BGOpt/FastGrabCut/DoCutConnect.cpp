#include "DoCutConnect.h"
#include<iostream>

DoCutConnect::DoCutConnect()
{
	Superpixels_Contour_Color = Scalar(0,0,0);
	Segment_Contour_Color = Scalar(255,255,255);
	Fore_Line_Color = Scalar(230,130,255);
	Fore_Pr_Line_Color = Scalar(255,0,0);
	Back_Line_Color = Scalar(255,255,160);
	Back_Pr_Line_Color = Scalar(0,255,0);
	Rect_Color = Scalar(0,0,255);

	isFirstPxlCut = true;
}

void DoCutConnect::setImage(Mat& inImage)
{
	image = inImage;
	mask.create(image.size(), CV_8UC1);
	mask.setTo(Scalar::all(GC_BGD));
}

void DoCutConnect::setColor(Scalar* fgColor, Scalar* bgColor, Scalar* ctColor)
{
	if(fgColor != NULL)
	{
		Fore_Pr_Line_Color = *fgColor;
	}
	if(bgColor != NULL)
	{
		Back_Line_Color = *bgColor;
	}
	if(ctColor != NULL)
	{
		Superpixels_Contour_Color = *ctColor;
	}
}

double DoCutConnect::doSuperpixel()
{
	double begin = (double)getTickCount();
	slico.DoSuperpixelSegmentation_ForGivenMat(image, klabels, numlabels);

	//////////////////////////////////   Lab
	Mat image2 = image.clone();
	cvtColor(image, image2, CV_BGR2Lab);

	slico.GetArcAndCenterOfSuperpixels(image2, klabels, numlabels, superarcs, superpixels, spContains);
	double end = (double)getTickCount();
	double time = (end-begin)/getTickFrequency();
	return time;
}

Mat DoCutConnect::doDrawArcAndOther()
{
	spRelation.assign(numlabels, -1);
	for(int i = 0; i < spRelation.size(); i++) spRelation[i] = i;

	spImage = image.clone();
	//slico.DrawAverageColor(spImage, klabels, superpixels);
	slico.DrawContoursAroundSegments(spImage, klabels, Superpixels_Contour_Color);
	return spImage;
}


void DoCutConnect::doSuperpixelSegmentation(Rect* _rect, set<MyPoint>* _fgdPixels, set<MyPoint>* _bgdPixels, set<MyPoint>* _prFgdPixels, set<MyPoint>* _prBgdPixels, int iterCount)
{

	if(_rect != NULL)		//初次分割
	{
		initSPMask(_rect);
		beta = spGrabcut.calcBeta(image);
		spGrabcut.grabCut_ForSuperpixels(klabels, superpixels, superarcs,  beta, spMask, fgdGMM, bgdGMM,1, GC_INIT_WITH_MASK);
		setFullMaskFromSPMask();
		prepareLaterGrabCut(superpixels, superarcs, spMask);	//修改标记为rect范围，并删减数据
	}else
	{
		updateSPMask(_fgdPixels, _bgdPixels, _prFgdPixels, _prBgdPixels);
		spGrabcut.grabCut_ForSuperpixels(klabels, superpixels, superarcs,  beta, spMask, fgdGMM, bgdGMM,1);
		setFullMaskFromSPMask();
	}


}


void DoCutConnect::doPixelSegmentation()
{

	//GC_BGD=0  GC_FGD=1  GC_PR_BGD=2  GC_PR_FGD=3
	Mat partImage;
	image.copyTo(partImage);

	fullMask.copyTo(mask);

	//image(rect).copyTo(partImage);
	//fullMask(rect).copyTo(mask);


	set<MyPoint>::iterator iter;
	for(iter = fgdPixels.begin(); iter != fgdPixels.end(); iter++)
	{
		if(rect.contains(Point(iter->x, iter->y)))
		{
			//int index = iter->y * image.cols + iter->x;
			int pr = iter->y - rect.y;
			int pc = iter->x - rect.x;
			mask.at<uchar>(pr,pc) = GC_FGD;
		}
	}

	for(iter = bgdPixels.begin(); iter != bgdPixels.end(); iter++)
	{
		if(rect.contains(Point(iter->x, iter->y)))
		{
			//int index = iter->y * image.cols + iter->x;
			int pr = iter->y - rect.y;
			int pc = iter->x - rect.x;
			mask.at<uchar>(pr,pc) = GC_BGD;
		}
	}

	for(iter = prFgdPixels.begin(); iter != prFgdPixels.end(); iter++)
	{
		if(rect.contains(Point(iter->x, iter->y)))
		{
			//int index = iter->y * image.cols + iter->x;
			int pr = iter->y - rect.y;
			int pc = iter->x - rect.x;
			mask.at<uchar>(pr,pc) = GC_PR_FGD;
		}
	}

	for(iter = prBgdPixels.begin(); iter != prBgdPixels.end(); iter++)
	{
		if(rect.contains(Point(iter->x, iter->y)))
		{
			//int index = iter->y * image.cols + iter->x;
			int pr = iter->y - rect.y;
			int pc = iter->x - rect.x;
			mask.at<uchar>(pr,pc) = GC_PR_BGD;
		}
	}

	if(isFirstPxlCut)
	{
		pxlGrabcut.grabCut_ForPixel(partImage, mask, rect, fgdGMM, bgdGMM, 1, true);
		isFirstPxlCut = false;
	}
	else
	{
		pxlGrabcut.grabCut_ForPixel(partImage, mask, rect, fgdGMM, bgdGMM, 1, false);
	}


	setFullMaskFromMask();
}



int DoCutConnect::getRealSPLabel(int pixelIndex)	//若返回-1，表示此像素在rect外面，所以超像素新标记没有
{
	return spRelation[klabels[pixelIndex]];
}


void DoCutConnect::initSPMask(Rect* _rect)
{
	rect = *_rect;
	spMask.assign(numlabels, GC_BGD);
	int width = image.cols;
	for(int r = 0; r < rect.height; r++)
	{
		for(int c = 0; c < rect.width; c++)
		{
			int index = (r+rect.y) * width + (c+rect.x);
			//int rIndex = r * rect.width + c;
			spMask[getRealSPLabel(index)] = GC_PR_FGD;
		}
	}
}
void DoCutConnect::updateSPMask(set<MyPoint>* _fgdPixels, set<MyPoint>* _bgdPixels, set<MyPoint>* _prFgdPixels, set<MyPoint>* _prBgdPixels)
{
	set<MyPoint>::iterator iter;
	if(_fgdPixels != NULL)
	{
		for(iter = _fgdPixels->begin(); iter != _fgdPixels->end(); iter++)
		{
			if(rect.contains(Point(iter->x, iter->y)))
			{
				int index = iter->y * image.cols + iter->x;
				int label = getRealSPLabel(index);
				if(label > -1) 
				{
					fgdPixels.insert(*iter);
					spMask[label] = GC_FGD;
				}
			}
		}
	}
	if(_bgdPixels != NULL)
	{
		for(iter = _bgdPixels->begin(); iter != _bgdPixels->end(); iter++)
		{
			if(rect.contains(Point(iter->x, iter->y)))
			{
				int index = iter->y * image.cols + iter->x;
				int label = getRealSPLabel(index);
				if(label > -1) 
				{
					bgdPixels.insert(*iter);
					spMask[label] = GC_BGD;
				}
			}
		}
	}
	if(_prFgdPixels != NULL)
	{
		for(iter = _prFgdPixels->begin(); iter != _prFgdPixels->end(); iter++)
		{
			if(rect.contains(Point(iter->x, iter->y)))
			{
				int index = iter->y * image.cols + iter->x;
				int label = getRealSPLabel(index);
				if(label > -1) 
				{
					prFgdPixels.insert(*iter);
					spMask[label] = GC_PR_FGD;
				}
			}
		}
	}
	if(_prBgdPixels != NULL)
	{
		for(iter = _prBgdPixels->begin(); iter != _prBgdPixels->end(); iter++)
		{
			if(rect.contains(Point(iter->x, iter->y)))
			{
				int index = iter->y * image.cols + iter->x;
				int label = getRealSPLabel(index);
				if(label > -1) 
				{
					prBgdPixels.insert(*iter);
					spMask[label] = GC_PR_BGD;
				}
			}
		}
	}


}




void DoCutConnect::prepareLaterGrabCut(vector<Vec6d>& superpixels, vector<vector<int> >& superarcs, vector<int>& spMask)
{

	//记录有那些超像素在Rect范围内。
	vector<bool> spUsed(spMask.size(), false);		
	int width = image.cols;

	for(int r = 0; r < rect.height; r++)
	{
		for(int c = 0; c < rect.width; c++)
		{
			int index = (r+rect.y) * width + (c+rect.x);
			//int rIndex = r * rect.width + c;
			spUsed[getRealSPLabel(index)] = true;
		}
	}

	int count = 0;
	for(int i = 0; i < spUsed.size(); i++)
	{
		if(spUsed[i]) count++;
	}


	//抛弃Rect外面的超像素，spRelation记录新旧标记的对应关系（spRelation[旧/整图]=spRelation[新]）。
	spRelation.assign(spMask.size(), -1);
	vector<Vec6d> superpixels2;
	int spSize = 0;
	vector<int> spMask2;
	vector<vector<int> > superarcs2;
	for(int i = 0; i < superpixels.size(); i++)
	{
		if(spUsed[i])
		{
			superpixels2.push_back(superpixels[i]);
			spRelation[i] = spSize;
			spSize++;

			//修改spMask
			spMask2.push_back(spMask[i]);		


			vector<int> temp;
			for(int j = 0; j < superarcs[i].size(); j++)
			{
				if(spUsed[superarcs[i][j]])
				{
					temp.push_back(superarcs[i][j]);
				}
			}
			superarcs2.push_back(temp);			//此时存的旧的超像素标记

		}
	}



	//更新边关系里的旧标记
	for(int i = 0; i < superarcs2.size(); i++)
	{
		for(int j = 0; j < superarcs2[i].size(); j++)
		{
			int k = spRelation[superarcs2[i][j]];		//修改老旧的超像素标记为新标记
			if(k == -1)
			{
				return;  //ERROR
			}
			superarcs2[i][j] = k;
		}
	}

	spNum = superpixels2.size();

	//替换superarcs, superpixels, spMask
	superarcs.clear();
	superpixels.clear();
	spMask.clear();

	superarcs.assign(superarcs2.begin(), superarcs2.end());
	superpixels.assign(superpixels2.begin(), superpixels2.end());
	spMask.assign(spMask2.begin(), spMask2.end());
}


Mat DoCutConnect::getImage(bool isSPImage)
{
	Mat binMask;
	Mat res;
	res.create(image.size(), image.type());
	res.setTo(Vec3b(255,255,255));				//设置空白区域颜色


    if( binMask.empty() || binMask.rows!=fullMask.rows || binMask.cols!=fullMask.cols )
        binMask.create( fullMask.size(), CV_8UC1 );
	binMask = fullMask & 1;			//以fullMask里内容的最低位（二进制）是0/1来决定，是0得到0，是1得到1

	if(isSPImage)
	{
		spImage.copyTo(res, binMask);
	}
	else 
	{
		image.copyTo(res, binMask);
	}

	return res;
}

Mat DoCutConnect::getFullMask()
{
	return fullMask;
}


void DoCutConnect::setFullMaskFromSPMask()
{

	fullMask.create(image.size(), CV_8UC1);
	fullMask.setTo(Scalar::all(GC_BGD));
	fullMask(rect).setTo(Scalar::all(GC_PR_FGD));


	vector<int> mapping(spMask.size(), 0);		//标记的新旧映射
	int count = 0;
	for(int i = 0; i < spRelation.size(); i++)
	{
		if(spRelation[i] > -1)
		{
			mapping[spRelation[i]] = i;
			count++;
		}
	}

	int heightR = fullMask.rows;
	int widthC = fullMask.cols;
	int bit = fullMask.channels();
	int width = image.cols;
	for(int i = 0; i < spMask.size(); i++)
	{
		int value = spMask[i];
		int orignalIndex = mapping[i];
		for(int j = 0; j < spContains[orignalIndex].size(); j++)
		{
			int index = spContains[orignalIndex][j];
			int r = index / width;
			int c = index % width;
			fullMask.at<uchar>(r,c) = value;
		}
	}
	
}

void DoCutConnect::setFullMaskFromMask()
{

	mask.copyTo(fullMask);

	//fullMask.create(image.size(), CV_8UC1);
	//fullMask.setTo(Scalar::all(GC_BGD));
	//mask.copyTo(fullMask(rect));

}


