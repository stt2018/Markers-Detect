//工程实现
//1、从txt文件列表中，依次读取单张图片进行检测
//2、依次输出图片中出现的靶标标签及其以图片中心点为坐标的中心坐标
//3、输出该靶标在所有图片中出现的总次数
//4、格式如下
//  1
// 出现靶标总数
// GCP000003 0.000000 0.000000 0.000000 2
//  靶标名称         靶标的三维坐标
//  3
// 出现总次数
// 0000  -323.000000 174.000000 1
// 0001  -288.889000 129.778000 1
// 0002  -222.000000 229.000000 1
// 图片名     靶标的中心坐标
// 创建时间 2018/6/27
// 创建人   单
// 最终修改时间 2018/6/27 17:00


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include<opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

namespace {
	const char* about = "Basic marker detection";
	const char* keys =
		"{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
		"DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
		"DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
		"DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16,"
		"DICT_APRILTAG_16h5=17, DICT_APRILTAG_25h9=18, DICT_APRILTAG_36h10=19, DICT_APRILTAG_36h11=20}"
		"{v        |       | Input from video file, if ommited, input comes from camera }"
		"{ci       | 0     | Camera id if input doesnt come from video (-v) }"
		"{c        |       | Camera intrinsic parameters. Needed for camera pose }"
		"{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
		"{dp       |       | File of marker detector parameters }"
		"{r        |       | show rejected candidates too }"
		"{refine   |       | Corner refinement: CORNER_REFINE_NONE=0, CORNER_REFINE_SUBPIX=1,"
		"CORNER_REFINE_CONTOUR=2, CORNER_REFINE_APRILTAG=3}";
}

/**
*/

static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["camera_matrix"] >> camMatrix;
	fs["distortion_coefficients"] >> distCoeffs;
	return true;
}


/**
*/
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
	fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
	fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
	fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
	fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
	fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
	fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
	fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
	fs["minDistanceToBorder"] >> params->minDistanceToBorder;
	fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
	fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
	fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
	fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
	fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
	fs["markerBorderBits"] >> params->markerBorderBits;
	fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
	fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
	fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
	fs["minOtsuStdDev"] >> params->minOtsuStdDev;
	fs["errorCorrectionRate"] >> params->errorCorrectionRate;
	return true;
}

//从txt文本中读取目标图片存储路径
int readImgLst(std::string szLst, std::vector<std::string> &vImgList)
{
	vImgList.clear();
	FILE *fp = fopen(szLst.c_str(), "r");
	if (fp)
	{
		char szBuf[1024];
		while (fgets(szBuf, 1024, fp))
		{
			if (szBuf[strlen(szBuf) - 1] == '\n')
			{
				szBuf[strlen(szBuf) - 1] = '\0';
			}
			if (strcmp(szBuf, "") == 0)
				continue;
			char szName[1024];
			sscanf(szBuf, "%s", szName);
			vImgList.push_back(szName);
		}
		fclose(fp);
	}
}//readImgLst

 //分析存储路径
static void str_splitpath(std::string strF, std::string &drive, std::string &dir,
	std::string &fname, std::string &ext)
{
	int nPos1 = strF.find_first_of(':');
	if (nPos1 >= 0)
	{
		drive = strF.substr(0, nPos1 + 1);
	}
	else
	{
		drive = "";
	}

	int nPos2 = (std::max)((int)(strF.find_last_of('\\')), (int)(strF.find_last_of('/')));
	if (nPos2 >= 0)
	{
		dir = strF.substr(nPos1 + 1, nPos2 - nPos1);
	}
	else
	{
		dir = "";
	}

	int nPos3 = strF.find_last_of('.');
	if (nPos3 >= 0)
	{
		fname = strF.substr(nPos2 + 1, nPos3 - nPos2 - 1);
		ext = strF.substr(nPos3, strF.length());
	}
	else
	{
		fname = strF.substr(nPos2 + 1, strF.length());
		ext = "";
	}
}//str_splitpath


 //求四边形几何中心点
Point2f CrossPoint(const Vec4f  line1, const Vec4f   line2)
{
	Point2f pt;
	float k1, k2, b1, b2;
	if (line1[0] == line1[2])
	{
		pt.x = line1[0];
		pt.y = line2[1] == line2[3] ? line2[1] :
			float(line2[1] - line2[3])*(pt.x - line2[0]) / (line2[0] - line2[2]) + line2[1];
	}

	else if (line2[0] == line2[2])
	{
		pt.x = line2[0];
		pt.y = line1[1] == line1[3] ? line1[1] :
			float(line1[1] - line1[3])*(pt.x - line1[0]) / (line1[0] - line1[2]) + line1[1];
	}
	else
	{
		k1 = float(line1[3] - line1[1]) / (line1[2] - line1[0]);
		b1 = float(line1[1] - k1*line1[0]);
		k2 = float(line2[3] - line2[1]) / (line2[2] - line2[0]);
		b2 = float(line2[1] - k2*line2[0]);
		pt.x = (b2 - b1) / (k1 - k2);
		pt.y = k1* pt.x + b1;
	}
	return pt;
}//CrossPoint

#if 0
 //人机交互中的鼠标事件
Mat org, dst, img, tmp;
void onMouse(int event, int x, int y, int flags, void *ustc)
//event鼠标事件，x.y鼠标坐标，flags拖拽和键盘操作代号
{
	static Point pre_pt = (-1, -1);//初始坐标  
	static Point cur_pt = (-1, -1);//实时坐标  
	char temp[16];
	if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆  
	{
		org.copyTo(img);//将原始图片复制到img中  
		sprintf_s(temp, "(%d,%d)", x, y);
		pre_pt = Point(x, y);
		putText(img, temp, pre_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255), 1, 8);//在窗口上显示坐标  
		circle(img, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);//划圆  
		cv::imshow("img", img);
	}
	else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))//左键没有按下的情况下鼠标移动的处理函数  
	{
		img.copyTo(tmp);//将img复制到临时图像tmp上，用于显示实时坐标  
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));//只是实时显示鼠标移动的坐标  
		cv::imshow("img", tmp);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))//左键按下时，鼠标移动，则在图像上划矩形  
	{
		img.copyTo(tmp);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
		rectangle(tmp, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);//在临时图像上实时显示鼠标拖动时形成的矩形  
		cv::imshow("img", tmp);
	}
	else if (event == CV_EVENT_LBUTTONUP)//左键松开，将在图像上划矩形  
	{
		org.copyTo(img);
		sprintf_s(temp, "(%d,%d)", x, y);
		cur_pt = Point(x, y);
		putText(img, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
		circle(img, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		rectangle(img, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);//根据初始点和结束点，将矩形画到img上  
		cv::imshow("img", img);
		img.copyTo(tmp);
		//截取矩形包围的图像，并保存到dst中  
		int width = abs(pre_pt.x - cur_pt.x);
		int height = abs(pre_pt.y - cur_pt.y);
		if (width == 0 || height == 0)
		{
			printf("width == 0 || height == 0");
			return;
		}
		dst = org(Rect(min(cur_pt.x, pre_pt.x), min(cur_pt.y, pre_pt.y), width, height));
		cv::namedWindow("dst", CV_WINDOW_NORMAL);
		cv::imshow("dst", dst);
		cv::waitKey(0);
	}
}
#endif

//找中心坐标，两种方法，1.f=true，平均值，2.f=false，几何中心
std::vector<cv::Point2f> FindCenter(std::vector<std::vector<cv::Point2f>> &corners, bool f)
{
	vector<Point2f> mcorners;
	if (f)
	{
		Point2f s = Point2f(0, 0);
		if (corners.size() <= 0)
		{
			printf("Markers Detecte Fail T_T");
		}
		else
		{
			for (int i = 0; i < corners.size(); i++)
			{
				s = Point2f(0, 0);
				int j = 0;
				while (j<corners[0].size())
				{
					s = s + corners[i][j];
					j++;
				}
				s = (s / 4);
				mcorners.push_back(s);
			}//for
		}
	}
	else
	{
		Vec4f l1corners(10), l2corners(10);
		if (corners.size() <= 0)
		{
			printf("Markers Detecte Fail T_T\n");
		}//if
		else
		{
			for (int i = 0; i < corners.size(); i++)
			{
				int n = 0;
				int t = 0;
				l1corners[t] = corners[i][n].x;
				l1corners[t + 1] = corners[i][n].y;
				l1corners[t + 2] = corners[i][n + 2].x;
				l1corners[t + 3] = corners[i][n + 2].y;
				l2corners[t] = corners[i][n + 1].x;
				l2corners[t + 1] = corners[i][n + 1].y;
				l2corners[t + 2] = corners[i][n + 3].x;
				l2corners[t + 3] = corners[i][n + 3].y;
				mcorners.push_back(CrossPoint(l1corners, l2corners));
			}
		}//else
	}
	return mcorners;

}//FindCenter


 //画出中心点
void drawCross(InputOutputArray dimg, vector<Point2f> dcorner, cv::Scalar s)
{
	if (dimg.empty())
	{
		printf("There is no input data !");
		return;
	}
	if (dcorner.size() <= 0)
	{
		printf("There is no center be found in this image!");
		return;
	}
	for (int i = 0; i < dcorner.size(); i++)
	{
		Point2f s1 = dcorner[i] - Point2f(0, 4);
		Point2f s2 = dcorner[i] + Point2f(0, 4);
		Point2f s3 = dcorner[i] - Point2f(4, 0);
		Point2f s4 = dcorner[i] + Point2f(4, 0);
		cv::line(dimg, s1, s2, s);
		cv::line(dimg, s3, s4, s);
	}//for
}//drawCross

 //定义两个储存结构体
typedef struct tagMarker
{
	int nImgID, nMarkId;//nImgID=图像名称？nMarkId=靶标名
	std::string szImgName;//szImgName=图像名 ，
	Point2d p;//p 靶标中心点
	int tagType;
	tagMarker()
	{
		tagType = 1;
	}
}SMarker;

typedef struct tagTiePt
{
	std::string szPtName;
	double xyz[3];
	int nType;
	std::vector<SMarker> vImgPts;
	tagTiePt()
	{
		szPtName = "s";
		memset(xyz, 0, 3 * sizeof(double));
		nType = 2;
	}
}STiePt;

//将检测数据按要求存储到txt文件中
void writeMaker(std::string szFile, std::vector<STiePt> &vTie)
{
	FILE *fp = fopen(szFile.c_str(), "w");
	if (fp)
	{
		fprintf(fp, " %d\n", vTie.size());
		for (int j = 0; j<vTie.size(); j++)
		{
			fprintf(fp, "%-s  %20.6lf %20.6lf %20.6lf %5d\n",
				vTie[j].szPtName.c_str(), vTie[j].xyz[0], vTie[j].xyz[1], vTie[j].xyz[2], vTie[j].nType);

			fprintf(fp, " %-5d \n", vTie[j].vImgPts.size());
			for (int i = 0; i < vTie[j].vImgPts.size(); i++)
			{
				SMarker ssMak = vTie[j].vImgPts[i];
				fprintf(fp, " %s %20.6lf %20.6lf %5d\n", ssMak.szImgName.c_str(), ssMak.p.x, ssMak.p.y, ssMak.tagType);
			}//for
		}//for
	}//if
	fclose(fp);
}//writeMaker

 //主函数
int main(int argc, char *argv[])
{
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);

	if (argc < 2)
	{
		parser.printMessage();
		return 0;
	}//if

	 /*--------------------------------------参数提取及检测模块-------------------------------------*/
	int dictionaryId = parser.get<int>("d");
	bool showRejected = parser.has("r");
	bool estimatePose = parser.has("c");
	float markerLength = parser.get<float>("l");

	Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
	if (parser.has("dp"))
	{
		bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
		if (!readOk)
		{
			cerr << "Invalid detector parameters file" << endl;
			return 0;
		}//if
	}//if

	if (parser.has("refine"))
	{
		//override cornerRefinementMethod read from config file
		detectorParams->cornerRefinementMethod = parser.get<int>("refine");
	}//if
	std::cout << "Corner refinement method (0: None, 1: Subpixel, 2:contour, 3: AprilTag 2): " << detectorParams->cornerRefinementMethod << std::endl;

	/*---------------------------------------字典选取模块-----------------------------------*/
	Ptr<aruco::Dictionary> dictionary =
		aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

	/*--------------------------------------存储路径读取模块----------------------------------*/
	vector <string> vImgList;
	readImgLst("imgtest1.txt", vImgList);
	int markerId = { 0 };
	int i = 0;
	InputArray camMatrix = 0;
	InputArray distCoeffs = 0;
	vector <vector<int>> id;//存储各图片中检测出的id
	vector<vector<Point2f>> center_out;
	std::vector<SMarker> vMarkers;
	std::vector<std::vector<STiePt>> vTiePt;

	//分析路径,找到图像名称,---这里要放在下面的循环外面
	std::string drive, dir, fname, ext;
	vector<string> nameList(vImgList.size());
	for (int n = 0; n < vImgList.size(); n++)
	{
		str_splitpath(vImgList[n], drive, dir, fname, ext);
		nameList[n] = fname; //不带扩展名
	}//for

	for (int i = 0; i < vImgList.size(); i++)
	{
		std::cout << "detecting image " << nameList[i] << std::endl;
		Mat image, imageCopy;

		/*--------------------------------图像数据读取模块---------------------------------------*/
		image = imread(vImgList[i], IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR);

		/*------------------------------------靶标识别模块---------------------------------------*/
		vector< int > ids;
		vector< vector< Point2f > > corners, rejected;

		// detect markers
		aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
		id.push_back(ids);


		// draw results
		image.copyTo(imageCopy);
		if (ids.size() > 0)
		{
			aruco::drawDetectedMarkers(imageCopy, corners);// , ids);

		}//if
		if (showRejected && rejected.size() > 0)
			aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));

		/*------------------------------------寻找靶标中心模块------------------------------------*/
		//找到中心
		std::vector<cv::Point2f>center = FindCenter(corners, 0);


		//中心点位置精确化
		//1.参数设置
		vector<Point2f> subcenter(center.size());
		Size winSize = Size(3, 3);
		Size zeroZone = Size(-1, -1);
		//精度或最大迭代数目，其中任意一个达到  迭代次数80，精度0.001
		TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 80, 0.001);

		//2.转灰度图
		cv::Mat imgGray;
		cv::cvtColor(image, imgGray, CV_BGR2GRAY);

		//3.如果中心点存在，精确它
		if (center.size() > 0)
		{
			cornerSubPix(imgGray, center, winSize, zeroZone, criteria);

		}//if

		 //存储图像信息（这里有待改进）
		for (int t = 0; t < center.size(); t++)
		{
			Point2f center_o;
			//
			int nImgWidth = image.size().width;
			int nImgHeight = image.size().height;
			center_o.x = center[t].x - nImgWidth / 2.0;
			center_o.y = -center[t].y + nImgHeight / 2.0;
			SMarker sMak;
			sMak.nImgID = i;
			sMak.nMarkId = ids[t];
			sMak.szImgName = nameList[i];
			sMak.p = center_o;
			vMarkers.push_back(sMak);
		}//for

		 /*------------------------------------------靶标中心确定模块---------------------------------------*/
		 /*思路：
		 1.以现存中心为中心提取2*2的感兴趣区域
		 2.遍历ROI区域像素
		 3.判断是否存在
		 两种情况：
		 a.一黑三白或一白三黑，统一为一个与其他不同
		 b,两个各自相同，需要对角位置上的像素值相同,否则不是
		 */
		subcenter.clear();
		if (center.size() > 0)
		{
			for (int i = 0; i < center.size(); i++)
			{
				Mat recImg = imgGray(Rect(center[i].x - 5, center[i].y - 5, 10, 10));
				threshold(recImg, recImg, 100, 255, CV_THRESH_BINARY);
				int val = recImg.at<uchar>(0, 0);//初值
				vector<Point2i> eqval, ieqval;//存储相等和不等的坐标
				Point2i p;
				for (int r = 0; r < recImg.rows; r++)
				{
					for (int c = 0; c < recImg.cols; c++)
					{
						p.x = r;
						p.y = c;
						if (recImg.at<uchar>(r, c) == val)
						{
							eqval.push_back(p);

						}//if
						else
						{
							ieqval.push_back(p);
						}//else
					}//for
				}//for
				if (eqval.size() == 1 || ieqval.size() == 1)
				{
					circle(imageCopy, center[i], 1, cv::Scalar(0, 255, 0), -1);
				}//if
				else if (eqval.size() == 2)
				{
					int a = eqval[1].x - eqval[0].x;
					int b = eqval[1].y - eqval[0].y;
					if (a == b)
						circle(imageCopy, center[i], 1, cv::Scalar(0, 255, 0), -1);
					else
						subcenter.push_back(center[i]);
				}//else if
				else
				{
					subcenter.push_back(center[i]);
				}//else
			}//for
		}//if
		Mat tmpImg, dstImg;
		imgGray.copyTo(tmpImg);
		vector<Point2f> tcenter;
		pyrDown(tmpImg, dstImg, Size(tmpImg.cols / 2, tmpImg.rows / 2));
		for (int l = 0; l < subcenter.size(); l++)
		{
			tcenter.push_back(subcenter[l] / 2);
		}//for
		cornerSubPix(dstImg, tcenter, Size(3, 3), zeroZone, criteria);
		vector<Point2f> scenter;
		for (int c = 0; c < tcenter.size(); c++)
		{
			scenter.push_back(tcenter[c] * 2);
		}//for
		drawCross(imageCopy, scenter, Scalar(200, 255, 0));

#if 0
		/*--------------------------------------人机交互模块--------------------------------------*/
		//鼠标获取
		Mat org = imageCopy;
		org.copyTo(img);
		org.copyTo(tmp);
		//定义一个img窗口 

		cv::namedWindow("img", CV_WINDOW_NORMAL);
		setMouseCallback("img", onMouse, 0);//调用回调函数
		cv::imshow("img", img);
		cv::waitKey(0);
		if (!dst.empty())
		{
			aruco::detectMarkers(dst, dictionary, corners, ids, detectorParams, rejected);
			Mat dstCopy;
			dst.copyTo(dstCopy);
			if (ids.size() > 0)
			{
				aruco::drawDetectedMarkers(dstCopy, corners, ids);
			}
			cv::namedWindow("imgdetec", CV_WINDOW_NORMAL);
			cv::imshow("imgdetec", dstCopy);
			//cv::waitKey(1);
			//dst = 0;
		}
#endif

		//保存已识别图片
		char outf[128];
		string out = "E:/ShanTing/stt/VSproject/detmak_out/test/confirm";
		sprintf_s(outf, "%s/con_%s.jpg", out.c_str(), nameList[i].c_str());
		imwrite(outf, imageCopy);
	}//for图像列表循环


	 /*----------------------------------------输出模块-------------------------------------*/
	std::cout << "orginzing tie points... " << std::endl;
	std::sort(vMarkers.begin(), vMarkers.end(), [](SMarker &a, SMarker &b)
	{
		return a.nMarkId < b.nMarkId;
	});
	//重组开始
	std::vector<STiePt> vTie;
	int flag = vMarkers[0].nMarkId;
	int t = 0;
	STiePt sTie, empty;
	for (int h = 0; h < vMarkers.size(); h++)
	{
		if (flag != vMarkers[h].nMarkId)
		{
			vTie.push_back(sTie);
			flag = vMarkers[h].nMarkId;
			t = t + 1;
			sTie = empty;
		}//if
		sTie.szPtName = "M_" + std::to_string(flag);
		sTie.vImgPts.push_back(vMarkers[h]);
	}
	vTie.push_back(sTie);
	//重组结束

	//写入开始
	std::cout << "writting gci... " << std::endl;
	writeMaker("E:\\ShanTing\\stt\\VSproject\\detect_out\\detect_out\\markerstest.gci", vTie);

	std::cout << getTickCount() << std::endl;
	return 0;
}//main