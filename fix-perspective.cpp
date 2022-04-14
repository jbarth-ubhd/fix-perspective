/* tabstop=4 */

#include <cstdio>
#include <iostream>
#include <limits>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/time.h>

/*	Perspektivkorrektur

	Verfahren:
	1. Bild invertieren, Hintergrund vom Bild abziehen.
	2. Bild in angle_step-Schritten von -angle_range bis +angle_range drehen
	3. Jeweils Zeilen- und Spaltensummen bilden
	4. Von Zeilen- und Spaltensummen "Hintergrund" abziehen
	5. Optimale Winkel mit max. Standardabweichung finden
		5a. für alle Winkel gleichzeitig
		5b. für links=rechts
		5c. für links und rechts getrennt
		5d. für oben=unten
		5e. für oben und unten getrennt
		Gefundene Winkel jeweils nur übernehmen, 
		falls sd(best) >= sd(median)*min_better.
	6. Winkel wurden am Rand gemessen, sollen aber nach innen versetzt sein 
		(kein weißen/transparenten Ränder).
		Daher Anpassung erforderlich (adapt_angles).
	7. Schnittpunkte der Winkelgeraden bestimmen
	8. Transformationsmatrix ausrechnen lassen
	9. Transformation durchführen.

	Annahmen:
		Für beste Ergebnisse: ge'crop'te Scans
		Halbwegs registerhaltiger Satz

	TODO: Auf Text-Struktur prüfen (etwa gleich große weiße und schwarze Bereiche)
	TODO: berücksichtigt opencv Dateiinternes rotate-flag?
	TODO: min_better für 5a schwächer, für 5c+5e härter?
*/	

#define MIN_TEXT_LINES 30.f
const float min_better=1.2f;
const float angle_range=7.0;
const float angle_step =0.1;

using namespace std;
using namespace cv;

int debug=0;
time_t init_sec;
int start_line[10];
float start[10];

#define STOP(depth) do { \
	float stop=mutime(); \
	printf("\t# %d) %4d..%4d: %.3f s %s\n", depth, start_line[depth], __LINE__, stop-start[depth], __func__); \
	start[depth]=stop; \
	start_line[depth]=__LINE__; \
} while(0)

void mutime_init(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	init_sec=tv.tv_sec;
	for(int i=0; i<10; i++) {
		start[i]=tv.tv_usec*1e-6;
		start_line[i]=0;
	}
}

float mutime(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec-init_sec)+tv.tv_usec*1e-6;
}

bool intersection(
	const Point2f M, const Point2f N, 
	const Point2f P, const Point2f Q,
	Point2f &R);

#define DIM_COL 0
#define DIM_ROW 1
#define SYNC true
#define INDEPENDENT false

// Mathematica-Definitionen
#define Power(a, b) powf(a, b)
#define Sec(x) (1.f/cosf(x))
#define RealAbs(x) fabsf(x)
#define Tan(x) tanf(x)
#define Pi M_PI
template <typename T> int sgn(T val) {
    // https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    return (T(0) < val) - (val < T(0));
}

/* Mathematica:
In[7]:= y[a_]=RealAbs[Tan[a*\[Pi]/180]]*h/2 -(a-b)/m
Out[7]= -((a-b)/m)+1/2 h RealAbs[Tan[(a \[Pi])/180]]
In[8]:= D[y[a], a]
Out[8]= -(1/m)+(h \[Pi] Sec[(a \[Pi])/180]^2 Tan[(a \[Pi])/180])/(360 RealAbs[Tan[(a \[Pi])/180]])
In[14]:= Out[8] //. {Tan[a*\[Pi]/180]->A}
Out[14]= -(1/m)+(A h \[Pi] Sec[(a \[Pi])/180]^2)/(360 RealAbs[A])
In[15]:= Out[14] //. {A/RealAbs[A]->Sign[A]}
Out[15]= -(1/m)+1/360 h \[Pi] Sec[(a \[Pi])/180]^2 Sign[A]
*/
float angle(float a, float h, float m, float b) {
	// Mathematica: y[a_] = RealAbs[Tan[a*\[Pi]/180]]*h/2 - (a - b)/m
	return fabsf(tanf(a*M_PI/180.f))*h/2.f -(a-b)/m;
}
float angle_s(float a, float h, float m) { // first derivative
	// CForm[D[y[a], a]]
	// → Tan(a*Pi)/180 / Realabs(Tan(a*Pi)/180) durch sgn(a) ersetzt
	return -(1.f/m) + (h*Pi*
      Power(Sec((a*Pi)/180.f),2.f)*
      sgn(a))/
    (360.f);
}

void adapt_angles(float &a0, float &a1, const float w, const float h) {
	// w & h → think col_sums
	float m0=(a1-a0)/w;
	float b0=a0;
	float m1=-m0;
	float b1=a1;

	if(fabsf(m0)<1e-7) m0=1e-7; // prevent div by zero
	if(fabsf(m1)<1e-7) m1=1e-7;

	if(debug>=10) printf("a0=%+11.8f           a1=%+11.8f\n", a0, a1);

	int n0=1;
	float p_a0;
	do {
		p_a0=a0;
		a0= a0 - angle(a0, h, m0, b0)/angle_s(a0, h, m0); // Newton's method
	} while(fabsf(a0-p_a0)>1e-6 && n0++<99);

	int n1=1;
	float p_a1;
	do {
		p_a1=a1;
		a1= a1 - angle(a1, h, m1, b1)/angle_s(a1, h, m1); // Newton's method
	} while(fabsf(a1-p_a1)>1e-6 && n1++<99);

	if(debug>=10) printf("a0=%+11.8f [%2d iter] a1=%+11.8f [%2d iter]\n", a0, n0, a1, n1);
}

float sqr(const float x) { return x*x; }
float calcSd(const Mat im, const int dim, float a0, float a1) {
	// "l" like "length"
	// dim: see reduce: 0=to single row, 1=to single column
    const int l_max=dim==DIM_COL ? im.size().width-1 : im.size().height-1;
    const float delta_a=(a1-a0)/l_max;

	float val[l_max+1];
	float mean=0;
    float float_a=a0 +.5; // +.5 wg. int()
	if(dim==DIM_COL) {
	    for(int l=0; l<=l_max; l++) {
			val[l]=im.at<uchar>(int(float_a), l           );
			mean+=val[l];
	        float_a+=delta_a;
		}
	} else { // DIM_ROW
	    for(int l=0; l<=l_max; l++) {
			val[l]=im.at<uchar>(l           , int(float_a));
			mean+=val[l];
	        float_a+=delta_a;
		}
	}
	mean/=l_max+1;

	float sd=0;
    for(int l=0; l<=l_max; l++) {
        sd+=sqr(val[l] - mean);
	}
	sd/=l_max; // nicht +1, Standardabweichung der Stichprobe
	sd=sqrt(sd);

	return sd;
}

int compar(const void *va, const void *vb) {
	float *a=(float *)va;
	float *b=(float *)vb;
	if(*a < *b) return -1;
	if(*a > *b) return  1;
	return 0;
}

void findBest(const Mat im, const int dim, bool sync, float &best_a0, float &best_a1, float &sd_max, float &sd_median) {
	if(debug) printf("--> dim=%d %s\n", dim, sync?"sync=true":"");
	if(debug && !sync) STOP(1);
	// "a" like "angle"
	// dim: see reduce: 0=to single row, 1=to single column
    sd_max=0.0;
	// try all angle combinations
    int a_max=dim==DIM_COL ? im.size().height-1 : im.size().width-1;

	long max_elem=(a_max+1) * (sync ? 1 : a_max+1);
	float sd_array[max_elem];
	int n_elem=0;

    for(int a0=0; a0<=a_max; a0++) {
		float a1_start=sync ? a0 : 0    ;
		float a1_stop =sync ? a0 : a_max;
        for(int a1=a1_start; a1<=a1_stop; a1++) {
            const float sd=calcSd(im, dim, a0, a1);
			sd_array[n_elem++]=sd;
            if(sd<=sd_max) continue;
            sd_max=sd;
            best_a0=a0;
            if(!sync) best_a1=a1;
        }
    }

	assert(n_elem==max_elem);
	qsort(sd_array, n_elem, sizeof(float), compar);
	sd_median=sd_array[n_elem/2];

	if(debug && !sync) STOP(1);
}

void findBest_lrtb(const Mat col_sums, const Mat row_sums, float &best_a, float &sd_max, float &sd_median) {
	// "a" like "angle"
	// dim: see reduce: 0=to single row, 1=to single column
    sd_max=0.0;
	// try all angle combinations
    int a_max=row_sums.size().width-1;
	float sd_array[a_max+1];

    for(int a=0; a<=a_max; a++) {
        const float sd_col=calcSd(col_sums, DIM_COL, a, a);
        const float sd_row=calcSd(row_sums, DIM_ROW, a, a);
		const float sd=(sd_col+sd_row)/2.; // Nicht vergleichbar mit findBest()
		sd_array[a]=sd;
        if(sd<=sd_max) continue;
        sd_max=sd;
        best_a=a;
    }

	qsort(sd_array, a_max+1, sizeof(float), compar);
	sd_median=sd_array[(a_max+1)/2];
}


int main(int argc, char **argv) {
	mutime_init();
	if(argc-1!=2) {
		printf("Fixes perspective of text images\n");
		printf("usage: %s input_image output_image\n", argv[0]);
		printf("e. g.  %s input.tif output.tif\n", argv[0]);
		return 1;
	}

    Mat im   =imread(argv[1], IMREAD_REDUCED_GRAYSCALE_2);
    if(im.data==NULL) return 255;
    im=255-im;
	if(debug) STOP(0);
    Mat blurM;
    int maxDim=std::max(im.size().width, im.size().height);

	/* opencv-4.x/modules/imgproc/src/smooth.dispatch.cpp
    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1; */
	float sigma=maxDim/MIN_TEXT_LINES;
	int ksize=sigma*2*2; if(! (ksize%2)) ksize++;
    // GaussianBlur(im, blurM, Size(ksize, ksize), sigma); // ist viel langsamer
	blur(im, blurM, Size(ksize, ksize));

    Mat wim=im-blurM;
	if(debug) imwrite("work.tif", wim);	

    int angle_steps=std::ceil(angle_range/angle_step)*2+1;
    int angle_off=(angle_steps-1)/2;
    float angle_factor=angle_range/angle_off;
    
    Mat row_sums=Mat(Size(0, im.size().height), CV_8UC1);
    Mat col_sums=Mat(Size(im.size().width , 0), CV_8UC1);
    for(int a=0; a<angle_steps; a++) {
        Mat rotM=getRotationMatrix2D(Point2f(im.size().width/2., im.size().height/2.), (a-angle_off)*angle_factor, 1.0);
        Mat rot;
        warpAffine(wim, rot, rotM, wim.size(), INTER_NEAREST);
    	Mat col_sum, row_sum;
	    reduce(rot, col_sum, DIM_COL, REDUCE_AVG, CV_8U);
        vconcat(col_sums, col_sum, col_sums);
    	reduce(rot, row_sum, DIM_ROW, REDUCE_AVG, CV_8U);
        hconcat(row_sums, row_sum, row_sums);
    }
	if(debug) STOP(0);

	Mat col_blur, row_blur;
	blur(col_sums, col_blur, Size(ksize, 3));
	if(debug>=3) imwrite("col_blur.tif", col_blur);
	col_sums-=col_blur;
	blur(row_sums, row_blur, Size(3, ksize));
	if(debug>=3) imwrite("row_blur.tif", row_blur);
	row_sums-=row_blur;

	float sd_max, sd_median;

	float floatmax=1000; // std::numeric_limits<float>::max();
	float a_lef=floatmax;
	float a_top=floatmax;
	float a_rig=floatmax;
	float a_bot=floatmax;

#define A_DEG(x) (((x)-angle_off)*angle_factor)

    float a_test1, a_test2, a_dummy;
    findBest_lrtb(col_sums, row_sums, a_test1, sd_max, sd_median);
	if(debug) printf("lrtb  =%+6.3f deg %f %f\n", A_DEG(a_test1), sd_max, sd_median);
	if(sd_max >= sd_median*min_better) {
		a_top=a_lef=a_rig=a_bot=A_DEG(a_test1);
		printf("%7.3f %7.3f l=%+6.3f r=%+6.3f t=%+6.3f b=%+6.3f\n", sd_max, sd_median, a_lef, a_rig, a_top, a_bot);
	}

    findBest(col_sums, DIM_COL, SYNC, a_test1, a_dummy, sd_max, sd_median);
    if(debug) printf("lr    =%+6.3f deg %f %f\n", A_DEG(a_test1), sd_max, sd_median);
	if(sd_max >= sd_median*min_better) {
		a_lef=a_rig=A_DEG(a_test1);
		printf("%7.3f %7.3f l=%+6.3f r=%+6.3f t=%+6.3f b=%+6.3f\n", sd_max, sd_median, a_lef, a_rig, a_top, a_bot);
	}

    findBest(col_sums, DIM_COL, INDEPENDENT, a_test1, a_test2, sd_max, sd_median);
    if(debug) printf("left  =%+6.3f right =%+6.3f deg %f %f\n", A_DEG(a_test1), A_DEG(a_test2), sd_max, sd_median);
	if(sd_max >= sd_median*min_better) {
		a_lef=A_DEG(a_test1); a_rig=A_DEG(a_test2);
		printf("%7.3f %7.3f l=%+6.3f r=%+6.3f t=%+6.3f b=%+6.3f\n", sd_max, sd_median, a_lef, a_rig, a_top, a_bot);
	}

	findBest(row_sums, DIM_ROW, SYNC, a_test1, a_dummy, sd_max, sd_median);
    if(debug) printf("tb    =%+6.3f deg %f %f\n", A_DEG(a_test1), sd_max, sd_median);
	if(sd_max >= sd_median*min_better) {
		a_top=a_bot=A_DEG(a_test1);
		printf("%7.3f %7.3f l=%+6.3f r=%+6.3f t=%+6.3f b=%+6.3f\n", sd_max, sd_median, a_lef, a_rig, a_top, a_bot);
	}

    findBest(row_sums, DIM_ROW, INDEPENDENT, a_test1, a_test2, sd_max, sd_median);
    if(debug) printf("top   =%+6.3f bottom=%+6.3f deg %f %f\n", A_DEG(a_test1), A_DEG(a_test2), sd_max, sd_median);
	if(sd_max >= sd_median*min_better) {
		a_top=A_DEG(a_test1); a_bot=A_DEG(a_test2);
		printf("%7.3f %7.3f l=%+6.3f r=%+6.3f t=%+6.3f b=%+6.3f\n", sd_max, sd_median, a_lef, a_rig, a_top, a_bot);
	}

	if(a_lef==floatmax || a_top==floatmax || a_rig==floatmax || a_bot==floatmax) {
		Mat rgbIm=imread(argv[1]);
		imwrite(argv[2], rgbIm);
		return 0;
	}

	// gemessen wurde am Rand, aber durch Verschiebung der "Winkelgeraden" nach innen
	// muss der Winkel angepasst werden.
	adapt_angles(a_lef, a_rig, im.size().width -1, im.size().height-1);
	adapt_angles(a_top, a_bot, im.size().height-1, im.size().width -1);

	if(debug) {
		double minVal, maxVal;
		minMaxIdx(col_sums, &minVal, &maxVal);
		col_sums*=255.f/maxVal;
		imwrite("colsums.tif", col_sums);
		minMaxIdx(row_sums, &minVal, &maxVal);
		row_sums*=255.f/maxVal;
		imwrite("rowsums.tif", row_sums);
	}

    Mat rgbIm=imread(argv[1]);
    int hm1=rgbIm.size().height-1;
    int wm1=rgbIm.size().width -1;

    float d_top=tan(-a_top*M_PI/180)*wm1;
    float d_bot=tan(-a_bot*M_PI/180)*wm1;
    float d_lef=tan(-a_lef*M_PI/180)*hm1; 
    float d_rig=tan(-a_rig*M_PI/180)*hm1; 

    Point2f top_P, top_Q;
    if(d_top>=0) {
        top_P=Point2f(0  ,  d_top); 
        top_Q=Point2f(wm1,  0 );
    } else {
        top_P=Point2f(0  ,  0 );
        top_Q=Point2f(wm1, -d_top);
    }
	if(debug) line(rgbIm, top_P, top_Q, Scalar(255,0,0), 3);

    Point2f bot_P, bot_Q;
    if(d_bot>=0) { 
        bot_P=Point2f(0  , hm1   ); 
        bot_Q=Point2f(wm1, hm1-d_bot);
    } else { 
        bot_P=Point2f(0  , hm1+d_bot); // + = -(-)
        bot_Q=Point2f(wm1, hm1   ); 
    }
	if(debug) line(rgbIm, bot_P, bot_Q, Scalar(255,0,0), 3);

    Point2f lef_P, lef_Q;
    if(d_lef>=0) {
        lef_P=Point2f( 0 , 0  ); 
        lef_Q=Point2f( d_lef, hm1);
    } else {
        lef_P=Point2f(-d_lef, 0  );
        lef_Q=Point2f( 0 , hm1);
    }
	if(debug) line(rgbIm, lef_P, lef_Q, Scalar(255,0,0), 3);

    Point2f rig_P, rig_Q;
    if(d_rig>=0) { 
        rig_P=Point2f(wm1-d_rig, 0  ); 
        rig_Q=Point2f(wm1   , hm1);
    } else { 
        rig_P=Point2f(wm1   , 0  );
        rig_Q=Point2f(wm1+d_rig, hm1); // + = -(-)
    }
	if(debug) line(rgbIm, rig_P, rig_Q, Scalar(255,0,0), 3);

	if(debug) imwrite("rgb-w-lines.tif", rgbIm);

    Point2f r;
    vector< Point2f> src_corners, dst_corners;

	// Reihenfolge GEGEN den Uhrzeigersinn (?)
    intersection(top_P, top_Q, lef_P, lef_Q, r);
    src_corners.push_back(r);
    dst_corners.push_back(Point2f(0,0));

    intersection(bot_P, bot_Q, lef_P, lef_Q, r);
    src_corners.push_back(r);
    dst_corners.push_back(Point2f(0,hm1));

    intersection(bot_P, bot_Q, rig_P, rig_Q, r);
    src_corners.push_back(r);
    dst_corners.push_back(Point2f(wm1,hm1));

    intersection(top_P, top_Q, rig_P, rig_Q, r);
    src_corners.push_back(r);
    dst_corners.push_back(Point2f(wm1,0));

    Mat M = getPerspectiveTransform(src_corners, dst_corners);
	cout << M << endl;
    warpPerspective(rgbIm, rgbIm, M, rgbIm.size());
    imwrite(argv[2], rgbIm);

    return 0;
}
