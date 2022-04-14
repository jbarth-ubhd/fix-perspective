#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace Eigen;

// Punkte werden in Zeichnungen mit großen Buchstaben gekennzeichnet.
bool intersection(
	const Point2f M, const Point2f N, 
	const Point2f P, const Point2f Q,
	Point2f &R)
{
	// Schnittpunkt Gerade M…N und P…Q

	Point2f NM=N-M;
	Point2f QP=Q-P;
	
	// M + NM*t = P + QP*v
	// NM*t - QP*v = P - M

	Point2f PM = P-M;

	Matrix2f A;
	Vector2f b;

	A(0,0)=NM.x; A(0,1)=-QP.x;
	A(1,0)=NM.y; A(1,1)=-QP.y;
	b(0)=PM.x;
	b(1)=PM.y;
	// file:///usr/share/doc/libeigen3-dev/html/group__TutorialLinearAlgebra.html
	Vector2f t_und_v = A.colPivHouseholderQr().solve(b);
	R = M + NM * t_und_v(0);
	return true;
}
