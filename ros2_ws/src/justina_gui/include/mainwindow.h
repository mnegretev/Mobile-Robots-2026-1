#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "rclcomm.h"
#include <iostream>
QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void btnFwdPressed();
    void btnFwdReleased();
    void btnBwdPressed();    
    void btnBwdReleased();
    void btnTurnLeftPressed();
    void btnTurnLeftReleased();
    void btnTurnRightPressed();
    void btnTurnRightReleased();

private:
    Ui::MainWindow *ui;
    RclComm *commNode;
};
#endif // MAINWINDOW_H
