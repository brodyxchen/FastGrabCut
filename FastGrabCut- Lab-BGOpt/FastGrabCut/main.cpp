#include "fastgrabcut.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	FastGrabCut w;
	w.show();
	return a.exec();
}
