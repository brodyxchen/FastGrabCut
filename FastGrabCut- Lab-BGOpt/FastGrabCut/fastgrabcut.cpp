#include "fastgrabcut.h"


//cvtColor(inImage, outImage, CV_BGR2Lab);

FastGrabCut::FastGrabCut(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
	resize(800,600);
	panel = new PaintPanel();
	scrollArea = new QScrollArea;
	scrollArea->setBackgroundRole(QPalette::Dark);
	scrollArea->setWidget(panel);
	scrollArea->widget()->setMinimumSize(600,600);

	setCentralWidget(scrollArea);
	IsFirstSPCut = true;
	IsFirstSave = true;
}

FastGrabCut::~FastGrabCut()
{
	delete panel;
	delete scrollArea;
}

void FastGrabCut::showImage(QImage& img)
{
	panel->setImage(img);
}
void FastGrabCut::showImage(Mat& mat)
{
	saveAsImage = mat.clone();
	Mat rgb;  
	QImage img;  
	if(mat.channels() == 3)    // RGB image  
	{  
		cvtColor(mat,rgb,CV_BGR2RGB);  
		img = QImage((const uchar*)(rgb.data),  //(const unsigned char*)  
			rgb.cols,rgb.rows,  
			rgb.cols*rgb.channels(),   //new add  
			QImage::Format_RGB888);  
	}else                     // gray image  
	{  
		img = QImage((const uchar*)(mat.data),  
			mat.cols,mat.rows,  
			mat.cols*mat.channels(),    //new add  
			QImage::Format_Indexed8);  
	}  



	//Mat img2 = img.clone();
	//cvtColor(img, img2, CV_BGR2RGB);
	//QImage qimg = QImage((unsigned char*)(img2.data), img2.cols, img2.rows, QImage::Format_RGB888);
	panel->setImage(img);
}

void FastGrabCut::reset()
{
	IsFirstSPCut = true;
	IsFirstSave = true;
}

void FastGrabCut::changeEvent(QEvent *e)
{
	QMainWindow::changeEvent(e);
	switch(e->type())
	{
	case QEvent::LanguageChange:
		ui.retranslateUi(this);
		break;
	default:
		break;
	}
}

void FastGrabCut::Open()
{
	originalPath = QFileDialog::getOpenFileName(this, tr("Open Image"), ".", tr("Image Files (*.png *.jpg *.bmp)"));

	if(originalPath != "")
	{
		string fileName = originalPath.toStdString();
		panel->setImage(originalPath);

		originalImage = imread(fileName);
		conn.setImage(originalImage);
	}
}
void FastGrabCut::Save()
{
	if(IsFirstSave)
	{
		SaveAs();
	}else
	{
		string stdPath = currentPath.toStdString();
		imwrite(stdPath, saveAsImage);
	}
	
	//QFileInfo fileInfo(QfileName);
	//QString suffix = fileInfo.suffix();
	//QString baseName = fileInfo.baseName();
	//QString absPath = fileInfo.absolutePath();

}
void FastGrabCut::SaveAs()
{
	QFileInfo oriFileInfo(originalPath);
	QString oriAbsPath = oriFileInfo.absolutePath();

	QString defaultPath = oriAbsPath + "/untitled.png";
	currentPath = QFileDialog::getSaveFileName(this, tr("Save File"), defaultPath, tr("Images (*.png)"));

	string stdPath = currentPath.toStdString();
	imwrite(stdPath, saveAsImage);

	IsFirstSave = false;
}
void FastGrabCut::selectForegroundColor(int index)
{
	//3
	Scalar color = Scalar(255, 255, 255);
	switch(index)
	{
	case 0: color = Scalar(255, 0, 0); break;
	case 1: color = Scalar(0, 255, 0); break;
	case 2: color = Scalar(0, 0, 255); break;
	case 3: color = Scalar(255, 255, 0); break;
	case 4: color = Scalar(255, 255, 255); break;
	case 5: color = Scalar(0, 0, 0); break;
	}
	QRgb qcolor = qRgb(color.val[0], color.val[1], color.val[2]);
	//panel->setColor(&qcolor, NULL, NULL);
	conn.setColor(&color, NULL, NULL);
}
void FastGrabCut::selectBackgroundColor(int index)
{
	//1
	Scalar color = Scalar(255, 255, 255);
	switch(index)
	{
	case 0: color = Scalar(255, 0, 0); break;
	case 1: color = Scalar(0, 255, 0); break;
	case 2: color = Scalar(0, 0, 255); break;
	case 3: color = Scalar(255, 255, 0); break;
	case 4: color = Scalar(255, 255, 255); break;
	case 5: color = Scalar(0, 0, 0); break;
	}
	QRgb qcolor = qRgb(color.val[0], color.val[1], color.val[2]);
	//panel->setColor(NULL, &qcolor, NULL);
	conn.setColor(NULL, &color, NULL);
}
void FastGrabCut::selectContourColor(int index)
{
	//2

	Scalar color = Scalar(255, 255, 255);
	switch(index)
	{
	case 0: color = Scalar(255, 0, 0); break;
	case 1: color = Scalar(0, 255, 0); break;
	case 2: color = Scalar(0, 0, 255); break;
	case 3: color = Scalar(255, 255, 0); break;
	case 4: color = Scalar(255, 255, 255); break;
	case 5: color = Scalar(0, 0, 0); break;
	}
	QRgb qcolor = qRgb(color.val[0], color.val[1], color.val[2]);
	//panel->setColor(NULL, NULL, &qcolor);
	conn.setColor(NULL, NULL, &color);
	
}
void FastGrabCut::Superpixels()
{
	double time = conn.doSuperpixel();
	Mat img = conn.doDrawArcAndOther();

	QString qtime = getFormatQTime(time);
	log(qtime, "Superpixels");
	showImage(img);

}
void FastGrabCut::SPCut()
{
	//double begin = (double)getTickCount();
	////...
	//double end = (double)getTickCount();
	//double time = (end-begin)/getTickFrequency();

	double begin = (double)getTickCount();
	double time = 0;
	if(IsFirstSPCut)
	{
		Rect rect;
		panel->getRect(rect);

		///////////////
		//rect = Rect(149,95,182,142);

		IsFirstSPCut = false;
		Rect* prect = &rect;
		conn.doSuperpixelSegmentation(prect, NULL, NULL, NULL, NULL, 1);
		double end = (double)getTickCount();
		time = (end-begin)/getTickFrequency();
		Mat res = conn.getImage(false);
		showImage(res);

	}else
	{
		vector<vector<Point> > foreLines, backLines;
		panel->getLines(foreLines, backLines);

		for(int i = 0; i < foreLines.size(); i++)
		{
			for(int j = 0; j < foreLines[i].size(); j++)
			{
				fgdPixels.insert(MyPoint(foreLines[i][j].x, foreLines[i][j].y));
			}
		}
		for(int i = 0; i < backLines.size(); i++)
		{
			for(int j = 0; j < backLines[i].size(); j++)
			{
				bgdPixels.insert(MyPoint(backLines[i][j].x, backLines[i][j].y));
			}
		}
		set<MyPoint>* pf = &fgdPixels;
		set<MyPoint>* pb = &bgdPixels;
		set<MyPoint>* ppf = &prFgdPixels;
		set<MyPoint>* ppb = &prBgdPixels;
		conn.doSuperpixelSegmentation(NULL, pf, pb, ppf, ppb, 1);

		double end = (double)getTickCount();
		time = (end-begin)/getTickFrequency();
		Mat res = conn.getImage(false);
		showImage(res);
	}
	QString qtime = getFormatQTime(time);
	log(qtime, "SPCut");

}
void FastGrabCut::PxlCut()
{
	double begin = (double)getTickCount();

	conn.doPixelSegmentation();
	double end = (double)getTickCount();
	double time = (end-begin)/getTickFrequency();
	QString qtime = getFormatQTime(time);
	log(qtime, "PxlCut");
	Mat res = conn.getImage(false);
	showImage(res);
	imshow("Original", originalImage);
}

QString FastGrabCut::getFormatQTime(double time)
{
	stringstream ss;
	ss<<time;
	string t;
	ss>>t;
	ss.clear();
	QString tt = QString::fromStdString(t);
	return tt;
}

void FastGrabCut::log(QString s, QString flag)
{
	ui.MessageShow->append(flag + " Time=" + s + "\n");
}