#include "PaintPanel.h"

#include <QString>  
#include <QMessageBox>  
#include <QPainter>  
#include <QPen>  
#include <QMouseEvent> 
#include <QApplication>


PaintPanel::PaintPanel()
{
	foreColor = qRgb(255,255,0);
	backColor = qRgb(0,255,0);
	contourColor = qRgb(0,0,255);
	SegmentColor = qRgb(255,255,255);
	rectColor = qRgb(0,255,0);
	

	hasImage = false;
	LeftPressed = false;
	RightPressed = false;
	CtrlPressed = false;

}

void PaintPanel::setImage(QString fileName)
{
	
	image.load(fileName);

	foreLines.clear();
	backLines.clear();

	hasImage = true;

	LeftPressed = false;
	RightPressed = false;
	CtrlPressed = false;
	update();

}

void PaintPanel::setImage(QImage& img)
{
	image = img.copy();

	foreLines.clear();
	backLines.clear();

	hasImage = true;
	LeftPressed = false;
	RightPressed = false;
	CtrlPressed = false;
	update();
}


void PaintPanel::setColor(QRgb* ForeColor, QRgb* BackColor, QRgb* ContourColor)
{
	if(ForeColor != NULL)
	{
		foreColor = *ForeColor;
	}
	if(BackColor != NULL)
	{
		backColor = *BackColor;
	}
	if(ContourColor != NULL)
	{
		contourColor = *ContourColor;
	}

	
}


void PaintPanel::paintEvent(QPaintEvent *)
{
	if(hasImage)
	{
		QPainter panelPainter(this);
		panelPainter.drawImage(0,0,image);
	}
}

void PaintPanel::mousePressEvent(QMouseEvent *event)
{
	if(!hasImage) return;
	LeftPressed = false;
	RightPressed = false;
	CtrlPressed = false;

	line.clear();
	startPoint = event->pos();

	if(event->button() == Qt::RightButton)
	{
		RightPressed = true;
		line.push_back(startPoint);
	}else if(event->button() == Qt::LeftButton && QApplication::keyboardModifiers() != Qt::ControlModifier)
	{
		LeftPressed = true;
		line.push_back(startPoint);
	}else if(event->button() == Qt::LeftButton && QApplication::keyboardModifiers() == Qt::ControlModifier)
	{
		LeftPressed = true;
		CtrlPressed = true;
	}

	

}

void PaintPanel::mouseMoveEvent(QMouseEvent *event)
{
	if(!hasImage) return;
	if(CtrlPressed && LeftPressed)
	{
		endPoint = event->pos();
		//paint();
	}else if(LeftPressed || RightPressed)
	{
		endPoint = event->pos();
		line.push_back(endPoint);
		paint();
	}
}

void PaintPanel::mouseReleaseEvent(QMouseEvent *event)
{
	if(!hasImage) return;
	endPoint == event->pos();
	paint();
	if(event->button() == Qt::RightButton)
	{
		RightPressed = false;
		line.push_back(endPoint);
		vector<QPoint> aLine(line);
		backLines.push_back(aLine);
	}else if(event->button() == Qt::LeftButton && QApplication::keyboardModifiers() != Qt::ControlModifier)
	{
		LeftPressed = false;
		line.push_back(endPoint);
		vector<QPoint> aLine(line);
		foreLines.push_back(aLine);
	}else if(event->button() == Qt::LeftButton && QApplication::keyboardModifiers() == Qt::ControlModifier)
	{
		LeftPressed = false;
		CtrlPressed = false;
	}
	

}

void PaintPanel::getLines(vector<vector<Point> >& aForeLines, vector<vector<Point> >& aBackLines)
{
	for(int i = 0; i < foreLines.size(); i++)
	{
		vector<Point> aline;
		for(int j = 0; j < foreLines[i].size(); j++)
		{
			aline.push_back(Point(foreLines[i][j].x(),foreLines[i][j].y()));
		}
		aForeLines.push_back(aline);
	}

	for(int i = 0; i < backLines.size(); i++)
	{
		vector<Point> aline;
		for(int j = 0; j < backLines[i].size(); j++)
		{
			aline.push_back(Point(backLines[i][j].x(),backLines[i][j].y()));
		}
		aBackLines.push_back(aline);
	}
}

void PaintPanel::getRect(Rect& aRect)
{
	aRect.x = rect.x();
	aRect.y = rect.y();
	aRect.width = rect.width();
	aRect.height = rect.height();
}

void PaintPanel::paint()
{
	if(CtrlPressed && LeftPressed)
	{
		int lx = startPoint.x() < endPoint.x() ? startPoint.x() : endPoint.x();
		int rx = startPoint.x() < endPoint.x() ? endPoint.x() : startPoint.x();
		int ty = startPoint.y() < endPoint.y() ? startPoint.y() : endPoint.y();
		int by = startPoint.y() < endPoint.y() ? endPoint.y() : startPoint.y();
		QRect aRect(lx, ty, rx-lx, by-ty);

		QPainter painter(&image);
		painter.setPen(rectColor);
		painter.drawRect(aRect);

		rect = aRect;
		update();
	}else if(LeftPressed)
	{
		QPainter painter(&image);
		painter.setPen(foreColor);
		painter.drawLine(startPoint, endPoint);

		startPoint = endPoint;
		update();
	}else if(RightPressed)
	{
		QPainter painter(&image);
		painter.setPen(backColor);
		painter.drawLine(startPoint, endPoint);

		startPoint = endPoint;
		update();
	}

	
}

