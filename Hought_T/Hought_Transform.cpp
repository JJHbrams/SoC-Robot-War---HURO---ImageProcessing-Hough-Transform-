/*
Subject : Hough Transform
Made by Mrjohd
Date 2017.07.13
Version 1.1.0
*/
// 디버깅 환경변수 PATH
// PATH=C:\Users\MrJohd\Desktop\Temporary back up\ETC\opencv\build\x64\vc14\bin;%PATH%
#ifdef _DEBUG 
#pragma comment (lib, "opencv_world310d.lib") 
#else 
#pragma comment (lib, "opencv_world310.lib")
#endif

#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

#define iwidth 390			// Image Width
#define iheight 122			// Image Height

#define TRUE 1
#define FALSE 0
#define PI 3.14

#define G_Threshold 159		// Gaussian filter threshold
#define Thres_High 110
#define Thres_Low  60		
#define Thres_Cut  30
#define Offset_Maxima	60
#define Non_Maxima		2
#define STRONG_Edge 255		// Strong Edge (Edges, which is larger than 'Thres_High')
#define WEAK_Edge 127		// Weak Edge (Edges, which is larger than 'Thres_Low' but smaller than 'Thres_High')

#define Theta	360			// Actual range : 0~180 degress, Gap is 0.5도
#define Rho_Max	817			// D​iagonal length of Given Image : 2* sqrt(iwidth*iwidth + iheight*iheight)
#define Vote_Thres	100		// Voting Threshold 

using namespace std;
using namespace cv;


// 전역변수 선언
unsigned int i; unsigned int j; unsigned int fi; unsigned int fj;	//	width index | hieght index | mask width per pixel index | mask width per pixel index
int num;	// Index for number of noise pixel
Mat img_temp(iheight, iwidth, CV_8UC1);								//	Image printing Temporary Variable

// Filtering Global Variable
// Gaussian
int inimg_G[iheight + 4][iwidth + 4] = { 0 };
int outimg_G[iheight + 4][iwidth + 4] = { 0 };
// Median
int inimg[iheight + 2][iwidth + 2] = { 0 };
int outimg[iheight + 2][iwidth + 2] = { 0 };

// EdgeDetecting Global Variable
int inimg_C[iheight + 2][iwidth + 2] = { 0 };
// Gradient value (Manhatton Length : G_mag = G_magX + G_magY)
float G_mag[iheight + 2][iwidth + 2] = { 0 };
// Gradient angle (Angle = arctan(G_magY / G_magX))
float Angle[iheight + 2][iwidth + 2] = { 0 };
// Hysteresis Edge Tracking Temporary variable
float Edge_temp[iheight][iwidth] = { 0 };

// Hough transform Global variable
float LOT_sin[Theta] = { 0 };
float LOT_cos[Theta] = { 0 };
int hough_cnt[Rho_Max][Theta] = { 0 };
Mat Hough_S(Rho_Max, Theta, CV_8UC1);								// Hough space printing temporary variable
int Hough_com[Rho_Max + 2][Theta + 2] = { 0 };

// User defined functions
void Salt_Pepper(Mat img)
{
	// Salt/Pepper Noise 발생 함수
	for (num = 0; num < 635; num++)
	{
		i = rand() % iheight;
		j = rand() % iwidth;
		int x = rand();
		if (x%iheight > iwidth / 2)
			img.at<uchar>(i, j) = 255;
		else
			img.at<uchar>(i, j) = 0;

	}
	imshow("Salt/Pepper", img);
}

void filter_median(Mat img)
{
	// Median filter 함수

	// 여백없이 필터링 (원본바깥은 0으로)
	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
			inimg[i][j] = img.at<uchar>(i - 1, j - 1);

	for (i = 1; i<iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
		{
			unsigned int cell[9];
			int t = 0;
			for (fi = 0; fi < 3; fi++)
			{
				for (fj = 0; fj < 3; fj++)
				{
					cell[t] = inimg[i - 1 + fi][j - 1 + fj];
					t++;
				}
			}

			while (1)
			{
				int flag = FALSE;
				for (int k = 0; k < 8; k++)
					if (cell[k + 1] > cell[k])
					{
						unsigned int temp;
						flag = TRUE;
						temp = cell[k];
						cell[k] = cell[k + 1];
						cell[k + 1] = temp;
					}
				if (flag == FALSE)	break;
			}
			outimg[i][j] = cell[4];
		}

	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
			img.at<uchar>(i - 1, j - 1) = outimg[i][j];

	imshow("Median", img);
}

void filter_Gaussian(Mat img)
{
	// Gaussian filter 함수
	const int G_mask[5][5] = { { 2, 4, 5, 4, 2 }
		,{ 4, 9, 12, 9, 4 }
		,{ 5, 12, 15, 12, 5 }
		,{ 4, 9, 12, 9, 4 }
	,{ 2, 4, 5, 4, 2 } };

	// 여백없이 필터링 (원본바깥은 0으로)
	for (i = 2; i < iheight + 2; i++)
		for (j = 2; j < iwidth + 2; j++)
			inimg_G[i][j] = img.at<uchar>(i - 2, j - 2);

	for (i = 2; i < iheight + 2; i++)
		for (j = 2; j < iwidth + 2; j++)
		{
			float value = 0;
			for (fi = 0; fi < 5; fi++)
			{
				for (fj = 0; fj < 5; fj++)
					value += G_mask[fi][fj] * inimg_G[i - 2 + fi][j - 2 + fj];
			}
			outimg_G[i][j] = value / G_Threshold;
		}
	for (i = 2; i < iheight + 2; i++)
		for (j = 2; j < iwidth + 2; j++)
			img.at<uchar>(i - 2, j - 2) = outimg_G[i][j];

	imshow("Gaussian", img);
}

void Canny_edge(Mat image)
{
	// 2. 에지 그래디언트 구하기[크기,방향]
	// Sobel mask 	
	const int S_maskX[3][3] = { { -1, 0, 1 }
							   ,{ -2, 0, 2 }
							   ,{ -1, 0, 1 } };

	const int S_maskY[3][3] = { { 1, 2, 1 }
						       ,{ 0, 0, 0 }
					           ,{ -1, -2, -1 } };
	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
			inimg_C[i][j] = image.at<uchar>(i - 1, j - 1);

	// Sobel Mask 적용
	for (i = 1; i < iheight + 1; i++)
	{
		for (j = 1; j < iwidth + 1; j++)
		{
			float G_magX = 0;									// 그래디언트 크기 - Horizontal
			float G_magY = 0;									// 그래디언트 크기 - Vertical
			for (fi = 0; fi< 3; fi++)
			{
				for (fj = 0; fj < 3; fj++)
				{
					G_magX += S_maskX[fi][fj] * inimg_C[i - 1 + fi][j - 1 + fj];
					G_magY += S_maskY[fi][fj] * inimg_C[i - 1 + fi][j - 1 + fj];
				}
			}

			Angle[i][j] = (180.0 / PI)*atan2((float)G_magY, (float)G_magX);				// 방향성위한 각도계산	

			if (Angle[i][j] > 180)	Angle[i][j] = -179.909;
			if (Angle[i][j] < 0)	Angle[i][j] = -Angle[i][j];
			//cout << Angle[i][j]<<" ";

			if (G_magX < 0)		G_magX = -G_magX;
			if (G_magY < 0)		G_magY = -G_magY;

			//G_mag[i][j] = sqrt(G_magX*G_magX + G_magY*G_magY);		// 유클리디안 거리 : 더정확, 느림
			G_mag[i][j] = G_magX + G_magY;								// 맨해튼 거리 : 근사값, 빠름

		}
	}

	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
		{
			// 4방향으로 변환 (0도 방향/ 45도 방향/ 90도 방향/ 135도 방향)

			if ((Angle[i][j] >= 0 && Angle[i][j] < 22.5) || (Angle[i][j] >= 157.5 && Angle[i][j] <= 180))
				Angle[i][j] = 0;

			if (Angle[i][j] >= 22.5 && Angle[i][j] < 67.5)
				Angle[i][j] = 45;

			if (Angle[i][j] >= 67.5 && Angle[i][j] < 112.5)
				Angle[i][j] = 90;

			if (Angle[i][j] >= 112.5 && Angle[i][j] < 157.5)
				Angle[i][j] = 135;

		}

	// G_mag의 최대 최소 & 데이터 스케일 변환
	int min = 10e10;
	int max = 10e-10;
	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
		{
			if (min > G_mag[i][j])	min = G_mag[i][j];
			if (max < G_mag[i][j])	max = G_mag[i][j];
		}

	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth + 1; j++)
		{
			G_mag[i][j] = (G_mag[i][j] - min)*255.0 / (float)(max - min);
			image.at<uchar>(i - 1, j - 1) = G_mag[i][j];
		}

	imshow("Sobel", image);
	/******************************** 3. Non-Maximum Surpression **************************************/
	// Local Maximum 이면 값을 유지
	for (i = 1; i < iheight + 1; i++)
		for (j = 1; j < iwidth  + 1; j++)
		{
			int Edge_dir = Angle[i][j];							// 현재 픽셀에서의 그래디언트 방향
			int Centre = G_mag[i][j];							// 현재 픽셀의 그래디언트 크기
			int UU = G_mag[i + 1][j];							// 현재 픽셀기준 위쪽 방향의 그래디언트 크기
			int DD = G_mag[i - 1][j];							// 현재 픽셀기준 아래쪽 방향의 그래디언트 크기
			int LL = G_mag[i][j - 1];							// 현재 픽셀기준 왼쪽 방향의 그래디언트 크기
			int RR = G_mag[i][j + 1];							// 현재 픽셀기준 오른쪽 방향의 그래디언트 크기
			int UL = G_mag[i + 1][j - 1];						// 현재 픽셀기준 위 왼쪽 방향의 그래디언트 크기
			int UR = G_mag[i + 1][j + 1];						// 현재 픽셀기준 위 오른쪽 방향의 그래디언트 크기
			int DL = G_mag[i - 1][j - 1];						// 현재 픽셀기준 아래 왼쪽 방향의 그래디언트 크기
			int DR = G_mag[i - 1][j + 1];						// 현재 픽셀기준 아래 오른쪽 방향의 그래디언트 크기

			if (Centre > Thres_Cut)
			{
				switch (Edge_dir)
				{
				case 0:
					if (((LL <= Centre) && (RR <= Centre)))			image.at<uchar>(i - 1, j - 1) += Offset_Maxima;
					if (!((LL <= Centre) && (RR <= Centre)))		image.at<uchar>(i - 1, j - 1) += Non_Maxima;
				case 45:
					if (((UR <= Centre) && (DL <= Centre)))			image.at<uchar>(i - 1, j - 1) += Offset_Maxima;
					if (!((UR <= Centre) && (DL <= Centre)))		image.at<uchar>(i - 1, j - 1) += Non_Maxima;
				case 90:
					if (((UU <= Centre) && (DD <= Centre)))			image.at<uchar>(i - 1, j - 1) += Offset_Maxima;
					if (!((UU <= Centre) && (DD <= Centre)))		image.at<uchar>(i - 1, j - 1) += Non_Maxima;
				case 135:
					if (((UL <= Centre) && (DR <= Centre)))			image.at<uchar>(i - 1, j - 1) += Offset_Maxima;
					if (!((UL <= Centre) && (DR <= Centre)))		image.at<uchar>(i - 1, j - 1) += Non_Maxima;
				}
			}
			else
				image.at<uchar>(i - 1, j - 1) = 0;

		}

	imshow("Non-Maximum Surpression", image);

	// 4. Hysteresis Edge Tracking
	// Double Threshold
	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
		{
			if (image.at<uchar>(i, j) >= Thres_High)	image.at<uchar>(i, j) = STRONG_Edge;
			if (image.at<uchar>(i, j) < Thres_High && image.at<uchar>(i, j) >= Thres_Low)	image.at<uchar>(i, j) = WEAK_Edge;
			if (image.at<uchar>(i, j) < Thres_Low)	image.at<uchar>(i, j) = 0;
		}
	imshow("Canny1", image);




	// Edge Tracking
	for (i = 1; i < iheight - 1; i++)
		for (j = 1; j < iwidth - 1; j++)
		{
			int Centre = image.at<uchar>(i, j);							// 현재 픽셀의 그래디언트 크기
			int UU = image.at<uchar>(i + 1, j);							// 현재 픽셀기준 위쪽 방향의 그래디언트 크기
			int DD = image.at<uchar>(i - 1, j);							// 현재 픽셀기준 아래쪽 방향의 그래디언트 크기
			int LL = image.at<uchar>(i, j - 1);							// 현재 픽셀기준 왼쪽 방향의 그래디언트 크기
			int RR = image.at<uchar>(i, j + 1);							// 현재 픽셀기준 오른쪽 방향의 그래디언트 크기
			int UL = image.at<uchar>(i + 1, j - 1);						// 현재 픽셀기준 위 왼쪽 방향의 그래디언트 크기
			int UR = image.at<uchar>(i + 1, j + 1);						// 현재 픽셀기준 위 오른쪽 방향의 그래디언트 크기
			int DL = image.at<uchar>(i - 1, j - 1);						// 현재 픽셀기준 아래 왼쪽 방향의 그래디언트 크기
			int DR = image.at<uchar>(i - 1, j + 1);						// 현재 픽셀기준 아래 오른쪽 방향의 그래디언트 크기

			if (Centre == STRONG_Edge)
			{
				Edge_temp[i][j] = STRONG_Edge;
				for (fi = 0; fi < 3; fi++)
					for (fj = 0; fj < 3; fj++)
						if (image.at<uchar>(i - 1 + fi, j - 1 + fj) > 0)
						{
							// 강한에지 근처의 약한에지들을 강한에지로 확정지음으로서 실제로 연관성이 있는 에지만 표현
							Edge_temp[i - 1 + fi][j - 1 + fj] = STRONG_Edge;
							//image.at<uchar>(i - 1 + fi, j - 1 + fj) = STRONG_Edge;
						}

			}

			if (Centre == WEAK_Edge)
			{
				// 주위 8개의 픽셀 중 하나라도 STRONG_Edge이면 STRONG_Edge로 변화
				if (UU == STRONG_Edge || DD == STRONG_Edge || LL == STRONG_Edge || RR == STRONG_Edge || UR == STRONG_Edge || UL == STRONG_Edge || DR == STRONG_Edge || DL == STRONG_Edge)
					Edge_temp[i][j] = STRONG_Edge;
				//image.at<uchar>(i, j) = STRONG_Edge;

				// 주위 8개의 픽셀 중 STRONG_Edge가 하나도 존재하지 않으면 0
				if (UU != STRONG_Edge && DD != STRONG_Edge && LL != STRONG_Edge && RR != STRONG_Edge && UR != STRONG_Edge && UL != STRONG_Edge && DR != STRONG_Edge && DL != STRONG_Edge)
					Edge_temp[i][j] = 0;
				//image.at<uchar>(i, j) = 0;
			}

		}

	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
			image.at<uchar>(i, j) = Edge_temp[i][j];

	imshow("Canny2", image);

	/**************************************************************************************************************************************************/
}

void LOT_angle()
{
	//Look Up Table (LOT) for each degrees (0~180)
	for (int ang = 0; ang < Theta; ang++)
	{
		LOT_sin[ang] = sin((PI / 360.0)*ang);
		LOT_cos[ang] = cos((PI / 360.0)*ang);
	}
}

void HoughT(Mat image, Mat tmp, int opt)		// Original image, Deformed image, option
{
	// initiating 'Hough_S'		
	for (i = 0; i < Rho_Max; i++)
		for (j = 0; j < Theta; j++)
			Hough_S.at<uchar>(i, j) = 0;


	// Look Up Table 
	LOT_angle();

	// initiating 'hough count'
	for (i = 0; i < Rho_Max; i++)
		for (j = 0; j < Theta; j++)
			hough_cnt[i][j] = 0;

	// hough count 
	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
			if (tmp.at<uchar>(i, j) != 0)
			{
				for (int m = 0; m < Theta; m++)
				{
					int R = (int)(i*LOT_sin[m] + j*LOT_cos[m] + 0.5);		// 0.5 는 반올림용
					//cout << R << "		"<< endl;
					if (R >= -(Rho_Max / 2) && R <= Rho_Max / 2)
					{
						R += Rho_Max / 2;
						// 겹치는 부분 누적하는 부분 ㅇㅈ?ㅇㅇㅈ~
						hough_cnt[R][m]++;
						// Hough Space
						if (opt == TRUE)
							Hough_S.at<uchar>(R, m)++;
						else
							Hough_S.at<uchar>(R, m) = 0;
					}
						
				}
			}
	imwrite("Hough_Space.jpg", Hough_S);
	imshow("Hough Space", Hough_S);

	// 선의 임계값설정 & 개수
	// Non-Maximum suppression
	for (i = 1; i < Rho_Max + 1; i++)
		for (j = 1; j < Theta + 1; j++)
			Hough_com[i][j] = hough_cnt[i - 1][j - 1];

	int t = 0;
	int comp = 0;

	for (i = 1; i < Rho_Max + 1; i++)
		for (j = 1; j < Theta + 1; j++)
		{
			int Centre = Hough_com[i][j];							// Center pixel's hough_cnt
			int cnt = 0;
			if (Centre > Vote_Thres)
			{
				for (fi = 0; fi < 3; fi++)
					for (fj = 0; fj < 3; fj++)
					{
						comp = Hough_com[i + fi - 1][j + fj - 1];
						if (!(Centre > comp))
							cnt += 1;
					}
				if (cnt > 1)
					hough_cnt[i - 1][j - 1] = 0;
			}

			else
				hough_cnt[i - 1][j - 1] = 0;
		}

	// Express Lines, remain not suppressed
	Mat Hough_S2(Rho_Max, Theta, CV_8UC1);
	t = 0;
	for (i = 0; i < Rho_Max; i++)
		for (j = 0; j < Theta; j++)
		{
			if (hough_cnt[i][j] != 0)	t++;
			Hough_S2.at<uchar>(i, j) = hough_cnt[i][j];
		}
			
	cout << "Number of Lines : " << t << endl;
	imshow("Hough Space2", Hough_S2);








	// 원본 이미지에 선 표현
	// Initiating Temporary Array 
	Mat fuck(iheight, iwidth, CV_8UC1);
	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
			fuck.at<uchar>(i, j) = 0;

	// Horizontal increament
	i = 0;
	int a = 0;
	for (j = 0; j < iwidth; j++)
		for (int m = 1; m < Theta; m++)
			for (int n = 0; n < Rho_Max; n++)
			{
				if (hough_cnt[n][m] != 0)
				{
					i = (unsigned int)((n - (Rho_Max / 2) - j*LOT_cos[m]) / LOT_sin[m]);
					if (i >= 0 && i < iheight)
					{
						image.at<uchar>(i, j) = 128;
						fuck.at<uchar>(i, j) = 255;
						a++;
						//for (int l = 0; l < 1000000; l++);
						//cout << a << "		" << i << endl;
					}			
					
				}
			}
	j = 0;
	// Vertical increament
	for (i = 0; i < iheight; i++)
		for (int m = 0; m < Theta; m++)
			for (int n = 0; n < Rho_Max; n++)
			{
				if (hough_cnt[n][m] != 0)
				{
					j = (unsigned int)((n - (Rho_Max / 2) - i*LOT_sin[m]) / LOT_cos[m]);
					if (j >= 0 && j < iwidth)
					{
						image.at<uchar>(i, j) = 128;
						fuck.at<uchar>(i, j) = 255;
						//cout << a << "		" << j << endl;
					}
						
				}
					
			}
	

	imshow("Result", image);
	imshow("Result_B", fuck);
}

void main()
{
	// Origianl image (Gray)
	Mat image = imread("s2.jpg", 0);
	imshow("Original Image", image);

	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
			img_temp.at<uchar>(i, j) = image.at<uchar>(i, j);
	
	// Salt Noise
	//Salt_Pepper(img_temp);

	/******************************** filtering **************************************/
	//filter_median(img_temp);				// 미디안 필터 (소금/후추 노이즈가 심할때)
	filter_Gaussian(img_temp);				// 가우시안 필터 (가우시안 노이즈(백색노이즈) 제거)

	/******************************** Edge detect **************************************/
	Canny_edge(img_temp);					// Canny Edge

	/******************************** Hough Transform **************************************/
	
	// Hough 테스트 샘플 이미지 생성
	Mat tmp(iheight, iwidth, CV_8UC1);
	for (i = 0; i < iheight; i++)
		for (j = 0; j < iwidth; j++)
		{
			tmp.at<uchar>(i, j) = 0;
			if (i == 20 && j == 100)
				tmp.at<uchar>(i, j) = 255;
			if (i == 100 && j == 45)
				tmp.at<uchar>(i, j) = 255;
			if (i == 200 && j == 100)
				tmp.at<uchar>(i, j) = 255;
		}
	//imshow("Hough Sample", tmp);
	
	HoughT(image, img_temp, 1);
		

	/**************************************************************************************/
	// 출력저장
	imwrite("TEST.jpg", image);

	cout << "Done!" << endl;
	// 3000ms 대기
	waitKey(30000);
}