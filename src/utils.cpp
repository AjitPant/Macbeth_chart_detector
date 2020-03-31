// Redistributions of source code must retain information about author of algrithm  reproduce as:
// A. Kordecki, H. Palus, Automatic detection of colour charts in images, Przegląd Elektrotechniczny, 90(9):197-202, 2014



#include "../include/utils.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;


const int MAXN=1e6;
int parent[MAXN];
int sz[MAXN];
void build()
{
    for(int i=0;i<MAXN;i++)
    {
        parent[i]=i;
        sz[i]=1;
    }
}
int find_set(int u)
{
    if(parent[u]==u)
        return u;
    return parent[u]=find_set(parent[u]);
}
void merge(int u, int v)
{
    u=find_set(u);
    v=find_set(v);
    if(u!=v)
    {
        if(sz[u]<sz[v])
            swap(u,v);
        parent[v]=u;
        sz[u]+=v;
    }
}
bool is_contour_good(std::vector<Point> contour)
{
    double area=contourArea(contour);
    double perimeter=arcLength(contour, true);
    double centricity= 4* CV_PI * area/ (perimeter*perimeter);
    return (0.65<=centricity)&&(centricity<=0.95);
}
Point2f contour_centroid(vector<Point> contour)
{
    Moments moment; 
    moment=moments(contour); 
    Point2f center={(moment.m10/moment.m00),(moment.m01/moment.m00)}; 
    return center;
}
void quantize_image(const Mat& src, Mat &dst, Mat & k_means_centers)
{
    Mat data;
    src.convertTo(data,CV_32F);
    data = data.reshape(1,data.total());

    // do kmeans
    Mat labels;
    kmeans(data, 25, labels, TermCriteria(TermCriteria::MAX_ITER, 10, 1), 3, 
           KMEANS_PP_CENTERS, k_means_centers);
    // reshape both to a single row of Vec3f pixels:
    k_means_centers = k_means_centers.reshape(3,k_means_centers.rows);
    data = data.reshape(3,data.rows);

    // replace pixel values with their center value:
    Vec3f *p = data.ptr<Vec3f>();
    for (size_t i=0; i<data.rows; i++) {
       int center_id = labels.at<int>(i);
       p[i] = k_means_centers.at<Vec3f>(center_id);
    }

    // back to 2d, and uchar:
    dst = data.reshape(3, src.rows);

}

void group_contours_area(const vector<vector<Point>>& src_contours, vector<vector<vector<Point>>> & grouped_contours )
{
    grouped_contours.clear();

    vector< pair<double, int> > contours;
    for(int i=0;i<src_contours.size();i++)
    {
        contours.push_back({contourArea(src_contours[i]), i});
    }
    sort(contours.begin(), contours.end());

    double meanArea=0.00001;
    double old_index=-1;
    for(int i=0;i<contours.size();i++)
    {
        if((-meanArea+contours[i].first)/meanArea<0.3 && (-meanArea+contours[i].first)/contours[i].first<0.3)
        {
            meanArea=(meanArea*(i-old_index)+contours[i].first)/(i-old_index+1);
        }
        else
        {
            grouped_contours.push_back(vector<vector<Point>>());
            meanArea=contours[i].first;
            old_index=i;
        }
        grouped_contours[grouped_contours.size()-1].push_back(src_contours[contours[i].second]);
    } 
}

void group_contours_by_coordinates(const vector<vector<vector<Point>>>& src_groups, vector<vector<vector<Point>>>& dst_contours)
{
    dst_contours.clear();

    for(int i=0;i<src_groups.size();i++)
    {
        build();
        for(int j=0;j<src_groups[i].size();j++) 
        {
            for(int k=j+1;k<src_groups[i].size();k++)
            {
    
                Point2f center_j=contour_centroid(src_groups[i][j]);                     
                Point2f center_k=contour_centroid(src_groups[i][k]);                     
                if(norm(center_j-center_k)<2*sqrt(min(contourArea(src_groups[i][j]), contourArea(src_groups[i][j]))))
                    merge(j,k);

            }
        }
        int visited[MAXN];
        memset(visited, -1, sizeof(visited));
        for(int j=0;j<src_groups[i].size();j++)
        {
            if(visited[find_set(j)]==-1)
            {
                dst_contours.push_back(vector<vector<Point>>());
                visited[find_set(j)]=dst_contours.size()-1;
            }
            dst_contours[visited[find_set(j)]].push_back(src_groups[i][j]);
        }


    }
}

void find_good_contours(const Mat& quantized_src,const Mat& k_means_centers, vector<vector<Point> > &good_contours)
{
    const int kernel_size=5;
    good_contours.clear();

    for(int i=0;i<k_means_centers.size().height;i++) 
    {
        Mat bin=(quantized_src==k_means_centers.at<Vec3f>(i));
        vector<Mat> channels(3);
        split(bin, channels);
        bin=channels[0];

        medianBlur(bin, bin, kernel_size);
        Mat structuringElement = getStructuringElement( MORPH_RECT, Size(kernel_size,kernel_size));
        morphologyEx( bin, bin, MORPH_CLOSE, structuringElement, Point(-1, -1), 1 );
        vector<vector<Point> > contours;
        findContours( bin, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );
        for(auto contour: contours)
        {

   

            if(is_contour_good(contour))
                good_contours.push_back(contour);
        }  
    }


}
bool is_patch_good(const vector<vector<Point>> & group, const Size2f sz)
{
    int height=sz.height;
    int width=sz.width;
    //for simplifying calculations assume height is less than width
    if(width<height)
        swap(width,height);
    double meanArea=0;
    for(auto contour:group)
    {
        meanArea+=contourArea(contour);
    }
    meanArea/=group.size();
    double meanSideLength=1.15*sqrt(meanArea);
    double rows=(height/meanSideLength);
    double cols=(width/meanSideLength);
    int n_blocks=4*6;

    if(! (0.5*n_blocks<=group.size() && group.size()<=1.25*n_blocks) )

        return false;
    if(! (3<rows && rows<7 ) )
        return false;
    if(! (5<cols && cols<8 ) )
        return false;
    return true;

}

void return_found_charts(const vector<vector<Point>>& group, vector<Point2f> &centers, int height, int width )
{
    centers.clear();

    if(!group.size())
        return;

    vector<Point> joined;

    for(auto i:group)
    {
          joined.insert(joined.end(), i.begin(), i.end());
    }

    RotatedRect minRect=minAreaRect(joined);

    Point2f vertices[4];
    minRect.points(vertices);
    vector<Point2f> vert;

    for(auto i:vertices)
    {
        vert.push_back(i);
    }

    if(!is_patch_good(group, minRect.size))
    {
        return ;
    }

  


    std::vector<Point2f> initvert={{0, 0}, {6, 0}, {6, 4}, {0, 4}};
    if(norm(vert[0]-vert[1])<norm(vert[1]-vert[2]))
        swap(vert[1], vert[3]);


    Mat H= findHomography(initvert, vert);
    if(H.empty())
        return;
    vector<Point2f > centers_of_patches;
    for(auto contour:group)
    {
        Point2f center=contour_centroid(contour);
        centers_of_patches.push_back(center);
    }
    vector<Point2f> centers_orig,centers_transformed;
    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            centers_orig.push_back({i+0.5, j+0.5});
        }
    }

    perspectiveTransform(centers_orig, centers_transformed, H);
    std::vector<Point2f> centers_orig_refined, centers_transformed_refined;
    for(int i=0;i<centers_orig.size();i++)
    {
        pair<double, Point2f> nearest_patch={std::numeric_limits<double>::max(), {-1, -1}};
        for(auto patch_center:centers_of_patches)
        {
           if(norm(centers_transformed[i]-patch_center)<nearest_patch.first)
               nearest_patch={norm(centers_transformed[i]-patch_center), patch_center}; 
        }
        if(nearest_patch.first<=20)
        {
            centers_orig_refined.push_back(centers_orig[i]),
            centers_transformed_refined.push_back(nearest_patch.second);
       
        }
    }
    if(centers_orig_refined.size()==0)
        return;
    H= findHomography(centers_orig_refined, centers_transformed_refined);
    if(H.empty())
        return;

    perspectiveTransform(centers_orig, centers, H);
    

}

void find_charts(const Mat& src, vector<vector<Point2f>> & chart_centers)
{
    chart_centers.clear();

    Mat k_means_centers, quantized_src;
    vector<vector<Point> > good_contours;
    quantize_image(src, quantized_src, k_means_centers);
    find_good_contours(quantized_src, k_means_centers,good_contours );



    vector<vector<vector<Point>>> grouped_by_area;
    group_contours_area(good_contours, grouped_by_area);

    vector<vector<vector<Point>> > grouped;
    group_contours_by_coordinates(grouped_by_area, grouped);

    for(int i=0;i<grouped.size();i++)
    {
        vector<Point2f> centers; 


        return_found_charts(grouped[i],centers, 6, 4);
        if(!centers.size())
            continue;
        chart_centers.push_back(centers);


    }
}


