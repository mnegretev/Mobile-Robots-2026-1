#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //ui->plainTextEdit->setPlainText("hello");
    this->commNode = new RclComm();
    QObject::connect(ui->btnFwd, SIGNAL(pressed()), this, SLOT(btnFwdPressed()));
    QObject::connect(ui->btnFwd, SIGNAL(released()), this, SLOT(btnFwdReleased()));
    QObject::connect(ui->btnBwd, SIGNAL(pressed()), this, SLOT(btnBwdPressed()));
    QObject::connect(ui->btnBwd, SIGNAL(released()), this, SLOT(btnBwdReleased()));
    QObject::connect(ui->btnTurnLeft, SIGNAL(pressed()), this, SLOT(btnTurnLeftPressed()));
    QObject::connect(ui->btnTurnLeft, SIGNAL(released()), this, SLOT(btnTurnLeftReleased()));
    QObject::connect(ui->btnTurnRight, SIGNAL(pressed()), this, SLOT(btnTurnRightPressed()));
    QObject::connect(ui->btnTurnRight, SIGNAL(released()), this, SLOT(btnTurnRightReleased()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::btnFwdPressed()
{
    //commNode->start_publishing_cmd_vel(0.3, 0, 0);
    commNode->publish_cmd_vel(0.3, 0, 0);
}

void MainWindow::btnFwdReleased()
{
    //commNode->stop_publishing_cmd_vel();
    commNode->publish_cmd_vel(0.0, 0, 0);
}

void MainWindow::btnBwdPressed()
{
    //qtRosNode->start_publishing_cmd_vel(-0.3, 0, 0);
    commNode->publish_cmd_vel(-0.3, 0, 0);
}

void MainWindow::btnBwdReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
    commNode->publish_cmd_vel(0, 0, 0);
}

void MainWindow::btnTurnLeftPressed()
{
    //qtRosNode->start_publishing_cmd_vel(0, 0, 0.5);
    commNode->publish_cmd_vel(0, 0, 0.5);
}

void MainWindow::btnTurnLeftReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
    commNode->publish_cmd_vel(0, 0, 0);
}

void MainWindow::btnTurnRightPressed()
{
    //qtRosNode->start_publishing_cmd_vel(0, 0, -0.5);
    commNode->publish_cmd_vel(0, 0, -0.5);
}

void MainWindow::btnTurnRightReleased()
{
    //qtRosNode->stop_publishing_cmd_vel();
    commNode->publish_cmd_vel(0, 0, 0);
}


void MainWindow::navBtnCalcPath_pressed()
{
}

void MainWindow::navBtnExecPath_pressed()
{
}

void MainWindow::armSbAnglesValueChanged(double d)
{
}

void MainWindow::armSbGripperValueChanged(double d)
{
}

void MainWindow::armTxtArticularGoalReturnPressed()
{
}

void MainWindow::armTxtCartesianGoalReturnPressed()
{
}

void MainWindow::armBtnXpPressed()
{
}

void MainWindow::armBtnXmPressed()
{
}

void MainWindow::armBtnYpPressed()
{
}

void MainWindow::armBtnYmPressed()
{
}

void MainWindow::armBtnZpPressed()
{
}

void MainWindow::armBtnZmPressed()
{
}

void MainWindow::armBtnRollpPressed()
{
}

void MainWindow::armBtnRollmPressed()
{
}

void MainWindow::armBtnPitchpPressed()
{
}

void MainWindow::armBtnPitchmPressed()
{
}

void MainWindow::armBtnYawpPressed()
{
}

void MainWindow::armBtnYawmPressed()
{
}

void MainWindow::arm_get_IK_and_update_ui(std::vector<double> cartesian)
{
}


void MainWindow::spgTxtSayReturnPressed()
{
}

void MainWindow::sprTxtFakeRecogReturnPressed()
{
}
