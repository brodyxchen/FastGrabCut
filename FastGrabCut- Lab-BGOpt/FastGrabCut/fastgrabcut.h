#ifndef FASTGRABCUT_H
#define FASTGRABCUT_H

#include <QtWidgets/QMainWindow>
#include <qfiledialog>
#include <qgraphicsview.h>
#include <qtabbar.h>
#include <qtabwidget.h>
#include <qcombobox.h>
#include <qscrollarea.h>

#include "stdafx.h"
#include "DoCutConnect.h"
#include "PaintPanel.h"
#include "ui_fastgrabcut.h"

class FastGrabCut : public QMainWindow
{
	Q_OBJECT

public:
	FastGrabCut(QWidget *parent = 0);
	~FastGrabCut();
	void showImage(QImage& img);
	void showImage(Mat& img);
	void reset();
	QString getFormatQTime(double time);
	void log(QString s, QString flag="No.0:");

protected:
	void changeEvent(QEvent *e);

private:
	Ui::FastGrabCutClass ui;
	DoCutConnect conn;

	set<MyPoint> fgdPixels, bgdPixels, prFgdPixels, prBgdPixels;
	
	PaintPanel *panel;
	QScrollArea *scrollArea;
	bool IsFirstSPCut;

	Mat originalImage;

	bool IsFirstSave;
	Mat saveAsImage;

	QString originalPath;
	QString currentPath;

private slots:
	void Open();
	void Save();
	void SaveAs();
	void selectForegroundColor(int);
	void selectBackgroundColor(int);
	void selectContourColor(int);
	void Superpixels();
	void SPCut();
	void PxlCut();
};

#endif // FASTGRABCUT_H
