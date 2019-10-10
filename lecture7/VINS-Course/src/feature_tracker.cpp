#include "feature_tracker.h"
#include <opencv2/core/types.hpp>
//#define harris

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE)
        mask = fisheye_mask.clone();
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    forw_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask.at<uchar>(it.second.first) == 255)
        {
            forw_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img;
    TicToc t_r;
    cur_time = _cur_time;

    if (EQUALIZE)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        //ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img;

    if (forw_img.empty())
    {
        prev_img = cur_img = forw_img = img;
    }
    else
    {
        forw_img = img;
    }

    forw_pts.clear();

    if (cur_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3);

        for (int i = 0; i < int(forw_pts.size()); i++)
            if (status[i] && !inBorder(forw_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF();
        //ROS_DEBUG("set mask begins");
        TicToc t_m;
#ifdef harris
        setMask();//非极大值抑制？？
        //ROS_DEBUG("set mask costs %fms", t_m.toc());

        //ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask);
        }
        else
            n_pts.clear();
#else

        //ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        //ROS_DEBUG("add feature begins");
        TicToc t_a;
////////////////////////////begin/////////////////////////////////////
        vector<cv::KeyPoint> kp;
        const float W=30;
        //fast特征点提取

        const int nDesiredFeatures = MAX_CNT - forw_pts.size();
        if(nDesiredFeatures>0)
        {
            const int nCols = int(COL/W);//21
            const int nRows = int(ROW/W);//16
            const int nCells = nCols*nRows;

            const int nFeatureEachCell = ceil(float(nDesiredFeatures)/nCells);
            vector<vector<vector<cv::KeyPoint> > > cellKeyPoints(nRows, vector<vector<cv::KeyPoint> >(nCols));

            //计算每个窗口的大小
            const int wCell = COL/nCols;//30
            const int hCell = ROW/nRows;//30

            vector<vector<int>> nTotal(nRows,vector<int>(nCols));
            vector<vector<bool>> bNoMore(nRows,vector<bool>(nCols,false));
            vector<vector<int>> nToRetain(nRows,vector<int>(nCols));
            int nNoMore = 0;
            int nToDistribute = 0;


            for(int i=0; i<nRows; i++)
            {
                //每个窗口纵向的范围
                const int iniY =i*hCell;
                int maxY = iniY+hCell;
                //出了图片的有效区域
//                if(iniY>=maxBorderY-3)
//                    continue;
                //超出了边界的话就使用图像的边界作为边界
//                if(maxY>maxBorderY)
//                    maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    //计算每列的位置
                    //每个窗口横向的范围
                    const int iniX =j*wCell;
                    int maxX = iniX+wCell;
                    //出了横向范围
//                    if(iniX>=maxBorderX-6)
//                        continue;
                    //超了横向范围，就使用图像的边界作为窗口的边界
//                    if(maxX>maxBorderX)
//                        maxX = maxBorderX;
                    //计算FAST关键点
                    //vector<cv::KeyPoint> vKeysCell;
                    //对每一个窗口都计算FAST角点
                    cellKeyPoints[i][j].reserve(nFeatureEachCell*5);
                    FAST(forw_img.rowRange(iniY,maxY).colRange(iniX,maxX),
                         cellKeyPoints[i][j],iniThFAST,true);

                    if(cellKeyPoints[i][j].size()<=3)
                    {
                        cellKeyPoints[i][j].clear();
                        FAST(forw_img.rowRange(iniY,maxY).colRange(iniX,maxX),cellKeyPoints[i][j],minThFAST,true);
                    }

                    //HarrisResponses(cellImage,cellKeyPoints[i][j], 7, HARRIS_K);
                    const int nKeys = cellKeyPoints[i][j].size();
                    nTotal[i][j] = nKeys;

                    if(nKeys>nFeatureEachCell)
                    {
                        nToRetain[i][j] = nFeatureEachCell;
                        bNoMore[i][j] = false;
                    }
                    else
                    {
                        nToRetain[i][j] = nKeys;
                        nToDistribute += nFeatureEachCell-nKeys;
                        bNoMore[i][j] = true;
                        nNoMore++;
                    }
                }
            }

            while(nToDistribute>0 && nNoMore<nCells)
            {
                int nNewFeaturesCell = nFeatureEachCell + ceil((float)nToDistribute/(nCells-nNoMore));
                nToDistribute = 0;

                for(int i=0; i< nRows; i++)
                {
                    for(int j=0; j<nCols; j++)
                    {
                        if(!bNoMore[i][j])
                        {
                            if(nTotal[i][j]>nNewFeaturesCell)
                            {
                                nToRetain[i][j] = nNewFeaturesCell;
                                bNoMore[i][j] = false;
                            }
                            else
                            {
                                nToRetain[i][j] = nTotal[i][j];
                                nToDistribute += nNewFeaturesCell-nTotal[i][j];
                                bNoMore[i][j] = true;
                                nNoMore++;
                            }
                        }
                    }
                }
            }

            keypoints.reserve(nDesiredFeatures*2);

            for(int i=0; i<nRows; i++)
            {
                for(int j=0; j<nCols; j++)
                {
                    vector<cv::KeyPoint> &keysCell = cellKeyPoints[i][j];
                    cv::KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                    if((int)keysCell.size()>nToRetain[i][j])
                        keysCell.resize(nToRetain[i][j]);

                    for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                    {
                        keysCell[k].pt.x+=j*wCell;
                        keysCell[k].pt.y+=i*hCell;
                        keysCell[k].size = 15;
                        keypoints.push_back(keysCell[k]);
                    }
                }
            }

            //排除提取到跟踪的特征点
            for(auto& k:keypoints){
                if(forw_pts.empty()){
                    vKeyPoints = keypoints;
                }
                else{
                    auto it = forw_pts.begin();
                    for(;it!=forw_pts.end();it++){
                        if(norm(*it - k.pt)<30) //追踪到的点半径为30的方位内不提取新的特征点
                            break;
                    }
                    if(it == forw_pts.end())
                        vKeyPoints.push_back(k);
                }
            }

            if((int)vKeyPoints.size()>nDesiredFeatures)
            {
                cv::KeyPointsFilter::retainBest(vKeyPoints,nDesiredFeatures);
                vKeyPoints.resize(nDesiredFeatures);
            }
        } else
            n_pts.clear();


#endif
        addPoints();
        //ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
#ifdef harris
        for (auto &p : n_pts) {
            forw_pts.push_back(p);
            ids.push_back(-1);
            track_cnt.push_back(1);
        }
#else
        for (auto &p : vKeyPoints) {
            forw_pts.push_back(p.pt);
            ids.push_back(-1);
            track_cnt.push_back(1);
        }
#endif
    }
    ////////////////end//////////////////////
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    cur_img = forw_img;
    cur_pts = forw_pts;
    undistortedPoints();
    prev_time = cur_time;
    keypoints.clear();
    vKeyPoints.clear();
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        //ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)//每个点进行校正
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);//这里有用相机模型进行过校正
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        //ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        //ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    cout << "reading paramerter of camera " << calib_file << endl;
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear();
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++)
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty())
    {
        double dt = cur_time - prev_time;
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)
        {
            if (ids[i] != -1)
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt;
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y));
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0));
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0));
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    prev_un_pts_map = cur_un_pts_map;
}
