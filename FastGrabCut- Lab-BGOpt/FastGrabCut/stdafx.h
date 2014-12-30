#ifndef _STDAFX_H_
#define _STDAFX_H_

#include <iostream>
#include <sstream>
#include <vector>
#include <hash_set>
#include <set>
#include <string>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <qdir.h>


using namespace std;
using namespace cv;

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

class MyPoint
{
public:
	int x;
	int y;
	MyPoint(int xx, int yy)
	{
		x = xx;
		y = yy;
	}
	bool operator<(const MyPoint& c) const
	{
		return ((x < c.x) || ((x == c.x) && (y < c.y) ) );
	}
};

#endif