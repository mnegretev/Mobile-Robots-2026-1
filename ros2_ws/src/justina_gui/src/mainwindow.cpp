#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //ui->plainTextEdit->setPlainText("hello");
    this->commNode = new RclComm();
    QObject::connect(ui->btnFwd, SIGNAL(pressed()), this, SLOT(btnFwdPressed()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::btnFwdPressed()
{
    commNode->start_publishing_cmd_vel(0.3, 0, 0);
}

void MainWindow::btnFwdReleased()
{
    commNode->stop_publishing_cmd_vel();
}

void MainWindow::btnBwdPressed()
{
    //qtRosNode->start_publishing_cmd_vel(-0.3, 0, 0);
}

void MainWindow::btnBwdReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnTurnLeftPressed()
{
    //qtRosNode->start_publishing_cmd_vel(0, 0, 0.5);
}

void MainWindow::btnTurnLeftReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
}

void MainWindow::btnTurnRightPressed()
{
    //qtRosNode->start_publishing_cmd_vel(0, 0, -0.5);
}

void MainWindow::btnTurnRightReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
}
