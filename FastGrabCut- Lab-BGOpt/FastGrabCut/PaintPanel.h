#pragma once

#include "stdafx.h"

#include <qwidget.h>
#include <qpoint.h>
#include <vector>


class PaintPanel: public QWidget
{
public:
    PaintPanel();  

	void setImage(QString fileName);
	void setImage(QImage& img);

	void setColor(QRgb* ForeColor, QRgb* BackColor, QRgb* ContourColor);
	void getLines(vector<vector<Point> >& aForeLines, vector<vector<Point> >& aBackLines);
	void getRect(Rect& aRect);

protected:
    void paintEvent(QPaintEvent* p);  
    void mousePressEvent(QMouseEvent *e);  
    void mouseMoveEvent(QMouseEvent *e);  
    void mouseReleaseEvent(QMouseEvent *e); 
	void paint();

private:
	QImage image;

	QRgb foreColor, backColor, contourColor, SegmentColor, rectColor;

	QPoint startPoint, endPoint;
	QRect rect;

	vector<QPoint> line;
	vector<vector<QPoint> > foreLines, backLines;

	bool hasImage;
	bool LeftPressed, RightPressed;
	bool CtrlPressed;
};