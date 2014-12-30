/********************************************************************************
** Form generated from reading UI file 'fastgrabcut.ui'
**
** Created by: Qt User Interface Compiler version 5.2.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FASTGRABCUT_H
#define UI_FASTGRABCUT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FastGrabCutClass
{
public:
    QWidget *centralWidget;
    QStatusBar *statusBar;
    QDockWidget *dockWidget;
    QWidget *dockWidgetContents;
    QTextEdit *MessageShow;
    QGroupBox *groupBox;
    QPushButton *openImage;
    QPushButton *SaveImage;
    QPushButton *SaveAsImage;
    QGroupBox *groupBox_2;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QComboBox *ForegroundColor;
    QComboBox *BackgroundColor;
    QComboBox *ContourColor;
    QGroupBox *groupBox_3;
    QPushButton *Superpixels;
    QPushButton *SuperpixelCut;
    QPushButton *PixelCut;
    QLabel *label_4;
    QLabel *label_5;
    QLabel *label_6;
    QLabel *label_7;

    void setupUi(QMainWindow *FastGrabCutClass)
    {
        if (FastGrabCutClass->objectName().isEmpty())
            FastGrabCutClass->setObjectName(QStringLiteral("FastGrabCutClass"));
        FastGrabCutClass->resize(1200, 594);
        centralWidget = new QWidget(FastGrabCutClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        FastGrabCutClass->setCentralWidget(centralWidget);
        statusBar = new QStatusBar(FastGrabCutClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        FastGrabCutClass->setStatusBar(statusBar);
        dockWidget = new QDockWidget(FastGrabCutClass);
        dockWidget->setObjectName(QStringLiteral("dockWidget"));
        dockWidget->setMinimumSize(QSize(200, 38));
        dockWidget->setMaximumSize(QSize(200, 524287));
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName(QStringLiteral("dockWidgetContents"));
        MessageShow = new QTextEdit(dockWidgetContents);
        MessageShow->setObjectName(QStringLiteral("MessageShow"));
        MessageShow->setGeometry(QRect(10, 400, 171, 151));
        groupBox = new QGroupBox(dockWidgetContents);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(10, 10, 171, 91));
        openImage = new QPushButton(groupBox);
        openImage->setObjectName(QStringLiteral("openImage"));
        openImage->setGeometry(QRect(20, 20, 75, 23));
        SaveImage = new QPushButton(groupBox);
        SaveImage->setObjectName(QStringLiteral("SaveImage"));
        SaveImage->setGeometry(QRect(20, 40, 75, 23));
        SaveAsImage = new QPushButton(groupBox);
        SaveAsImage->setObjectName(QStringLiteral("SaveAsImage"));
        SaveAsImage->setGeometry(QRect(20, 60, 75, 23));
        groupBox_2 = new QGroupBox(dockWidgetContents);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 100, 171, 111));
        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(10, 20, 41, 16));
        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 50, 41, 16));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(10, 80, 41, 16));
        ForegroundColor = new QComboBox(groupBox_2);
        ForegroundColor->setObjectName(QStringLiteral("ForegroundColor"));
        ForegroundColor->setGeometry(QRect(60, 20, 61, 22));
        BackgroundColor = new QComboBox(groupBox_2);
        BackgroundColor->setObjectName(QStringLiteral("BackgroundColor"));
        BackgroundColor->setGeometry(QRect(60, 50, 61, 22));
        ContourColor = new QComboBox(groupBox_2);
        ContourColor->setObjectName(QStringLiteral("ContourColor"));
        ContourColor->setGeometry(QRect(60, 80, 61, 22));
        groupBox_3 = new QGroupBox(dockWidgetContents);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(10, 210, 171, 121));
        Superpixels = new QPushButton(groupBox_3);
        Superpixels->setObjectName(QStringLiteral("Superpixels"));
        Superpixels->setGeometry(QRect(10, 20, 101, 23));
        SuperpixelCut = new QPushButton(groupBox_3);
        SuperpixelCut->setObjectName(QStringLiteral("SuperpixelCut"));
        SuperpixelCut->setGeometry(QRect(10, 70, 101, 23));
        PixelCut = new QPushButton(groupBox_3);
        PixelCut->setObjectName(QStringLiteral("PixelCut"));
        PixelCut->setGeometry(QRect(10, 90, 101, 23));
        label_4 = new QLabel(groupBox_3);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(20, 50, 91, 16));
        label_5 = new QLabel(dockWidgetContents);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(20, 340, 91, 20));
        label_6 = new QLabel(dockWidgetContents);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(20, 360, 91, 20));
        label_7 = new QLabel(dockWidgetContents);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(20, 380, 91, 20));
        dockWidget->setWidget(dockWidgetContents);
        FastGrabCutClass->addDockWidget(static_cast<Qt::DockWidgetArea>(1), dockWidget);

        retranslateUi(FastGrabCutClass);
        QObject::connect(openImage, SIGNAL(clicked()), FastGrabCutClass, SLOT(Open()));
        QObject::connect(SaveImage, SIGNAL(clicked()), FastGrabCutClass, SLOT(Save()));
        QObject::connect(SaveAsImage, SIGNAL(clicked()), FastGrabCutClass, SLOT(SaveAs()));
        QObject::connect(ForegroundColor, SIGNAL(currentIndexChanged(int)), FastGrabCutClass, SLOT(selectForegroundColor(int)));
        QObject::connect(BackgroundColor, SIGNAL(currentIndexChanged(int)), FastGrabCutClass, SLOT(selectBackgroundColor(int)));
        QObject::connect(ContourColor, SIGNAL(currentIndexChanged(int)), FastGrabCutClass, SLOT(selectContourColor(int)));
        QObject::connect(Superpixels, SIGNAL(clicked()), FastGrabCutClass, SLOT(Superpixels()));
        QObject::connect(SuperpixelCut, SIGNAL(clicked()), FastGrabCutClass, SLOT(SPCut()));
        QObject::connect(PixelCut, SIGNAL(clicked()), FastGrabCutClass, SLOT(PxlCut()));

        ForegroundColor->setCurrentIndex(3);
        BackgroundColor->setCurrentIndex(2);
        ContourColor->setCurrentIndex(4);


        QMetaObject::connectSlotsByName(FastGrabCutClass);
    } // setupUi

    void retranslateUi(QMainWindow *FastGrabCutClass)
    {
        FastGrabCutClass->setWindowTitle(QApplication::translate("FastGrabCutClass", "FastGrabCut", 0));
        dockWidget->setWindowTitle(QApplication::translate("FastGrabCutClass", "\345\267\245\345\205\267", 0));
        groupBox->setTitle(QApplication::translate("FastGrabCutClass", "\345\233\276\347\211\207\346\223\215\344\275\234", 0));
        openImage->setText(QApplication::translate("FastGrabCutClass", "Open", 0));
        SaveImage->setText(QApplication::translate("FastGrabCutClass", "Save", 0));
        SaveAsImage->setText(QApplication::translate("FastGrabCutClass", "SaveAs", 0));
        groupBox_2->setTitle(QApplication::translate("FastGrabCutClass", "GroupBox", 0));
        label->setText(QApplication::translate("FastGrabCutClass", "Fore", 0));
        label_2->setText(QApplication::translate("FastGrabCutClass", "Back", 0));
        label_3->setText(QApplication::translate("FastGrabCutClass", "Contour", 0));
        ForegroundColor->clear();
        ForegroundColor->insertItems(0, QStringList()
         << QApplication::translate("FastGrabCutClass", "Red", 0)
         << QApplication::translate("FastGrabCutClass", "Green", 0)
         << QApplication::translate("FastGrabCutClass", "Blue", 0)
         << QApplication::translate("FastGrabCutClass", "Yellow", 0)
         << QApplication::translate("FastGrabCutClass", "White", 0)
         << QApplication::translate("FastGrabCutClass", "Black", 0)
        );
        BackgroundColor->clear();
        BackgroundColor->insertItems(0, QStringList()
         << QApplication::translate("FastGrabCutClass", "Red", 0)
         << QApplication::translate("FastGrabCutClass", "Green", 0)
         << QApplication::translate("FastGrabCutClass", "Blue", 0)
         << QApplication::translate("FastGrabCutClass", "Yellow", 0)
         << QApplication::translate("FastGrabCutClass", "White", 0)
         << QApplication::translate("FastGrabCutClass", "Black", 0)
        );
        ContourColor->clear();
        ContourColor->insertItems(0, QStringList()
         << QApplication::translate("FastGrabCutClass", "Red", 0)
         << QApplication::translate("FastGrabCutClass", "Green", 0)
         << QApplication::translate("FastGrabCutClass", "Blue", 0)
         << QApplication::translate("FastGrabCutClass", "Yellow", 0)
         << QApplication::translate("FastGrabCutClass", "White", 0)
         << QApplication::translate("FastGrabCutClass", "Black", 0)
        );
        groupBox_3->setTitle(QApplication::translate("FastGrabCutClass", "GroupBox", 0));
        Superpixels->setText(QApplication::translate("FastGrabCutClass", "Superpixels", 0));
        SuperpixelCut->setText(QApplication::translate("FastGrabCutClass", "SPCut", 0));
        PixelCut->setText(QApplication::translate("FastGrabCutClass", "PxlCut", 0));
        label_4->setText(QApplication::translate("FastGrabCutClass", "\347\224\273\346\241\206", 0));
        label_5->setText(QApplication::translate("FastGrabCutClass", "Ctrl+Left=Rect", 0));
        label_6->setText(QApplication::translate("FastGrabCutClass", "Left=ForeLine", 0));
        label_7->setText(QApplication::translate("FastGrabCutClass", "Right=BackLine", 0));
    } // retranslateUi

};

namespace Ui {
    class FastGrabCutClass: public Ui_FastGrabCutClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FASTGRABCUT_H
