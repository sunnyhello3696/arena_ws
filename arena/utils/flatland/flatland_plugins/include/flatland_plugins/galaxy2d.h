/*
 * @Author: sunnyhello369 sunnyhello369@gmail.com
 * @Date: 2023-03-14 20:55:15
 * @LastEditors: sunnyhello369 sunnyhello369@gmail.com
 * @LastEditTime: 2023-03-16 12:16:20
 * @FilePath: \230228junior\src\program\230314py2cpp\galaxy2d.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <limits>
#include <numeric>
#include <geometry_msgs/Polygon.h>
#include <geometry_msgs/PolygonStamped.h>

#ifndef GALAXY2D_H
#define GALAXY2D_H


typedef std::pair<double, double> Point;
typedef std::vector<Point> Points;
typedef std::pair<Points, std::vector<int>> Result;
typedef std::pair<Points, std::vector<double>> Result2;
typedef std::tuple<geometry_msgs::Polygon,std::vector<float>, std::vector<float>> Result3;

void polar2xy(double center_x, double center_y, double center_yaw, std::vector<double>& scan,
              std::vector<double>& x, std::vector<double>& y);

double NormalizeAngle(double d_theta);

double NormalizeAngleTo2Pi(double d_theta);

double cal_cross_product(const Point& p1, const Point& p2, const Point& p3);

bool Graham_Andrew_Scan(Result& res,const Points& points);

void sparse_scan(Result2& res,const Points& scans_xy, double drad);

bool is_in_convex(const Point& p,const Points& convex,bool is_clock_wise);

bool galaxy_xyin_360out(Result3& res,const Points& scans_xy,int max_vertex_num,double origin_x, double origin_y, double radius);

void Q_equitable_distribution(std::vector<int>& res,std::vector<double>& arr,int n);

#endif





