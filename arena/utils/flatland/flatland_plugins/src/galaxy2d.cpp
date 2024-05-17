#include "flatland_plugins/galaxy2d.h"
#include <ros/ros.h>

void polar2xy(double center_x, double center_y, double center_yaw, std::vector<double>& scan,
              std::vector<double>& x, std::vector<double>& y) {
    double drad = 1.0 / scan.size() * 2.0 * M_PI;
    x.clear();
    y.clear();
    for (int i = 0; i < scan.size(); i++) {
        x.push_back(center_x + scan[i] * std::cos(center_yaw + i * drad));
        y.push_back(center_y + scan[i] * std::sin(center_yaw + i * drad));
    }
}

double NormalizeAngle(double d_theta) {
    double d_theta_normalize = d_theta;
    while (d_theta_normalize > M_PI) {
        d_theta_normalize = d_theta_normalize - 2 * M_PI;
    }
    while (d_theta_normalize < -M_PI) {
        d_theta_normalize = d_theta_normalize + 2 * M_PI;
    }
    return d_theta_normalize;
}

double NormalizeAngleTo2Pi(double d_theta) {
    double d_theta_normalize = d_theta;

    while (d_theta_normalize > 2 * M_PI) {
        d_theta_normalize = d_theta_normalize - 2 * M_PI;
    }

    while (d_theta_normalize < 0) {
        d_theta_normalize = d_theta_normalize + 2 * M_PI;
    }
    return d_theta_normalize;
}


// Function to evenly distribute points in polar coordinates using linear interpolation
bool evenlyDistributePoints(const std::vector<float>& polar_convex,
                            const std::vector<float>& polar_convex_theta,
                            std::vector<float>& evenly_polar_convex,
                            std::vector<float>& evenly_polar_convex_theta,
                            int num_points) {
    if (polar_convex.size() != polar_convex_theta.size()) {
        ROS_ERROR("Input vectors must have the same size.");
        return false;
    }
    
    if (num_points <= 0) {
        ROS_ERROR("Number of points must be positive.");
        return false;
    }
    
    int n = polar_convex.size();
    float angle_step = 2 * M_PI / num_points;

    evenly_polar_convex.clear();
    evenly_polar_convex_theta.clear();

    for (int i = 0; i < num_points; ++i) {
        float target_angle = i * angle_step;

        // Find the two closest points in the original data
        float angle_diff_min = std::numeric_limits<float>::max();
        int closest_index_1 = -1;
        int closest_index_2 = -1;
        for (int j = 0; j < n; ++j) {
            float angle_diff = std::abs(polar_convex_theta[j] - target_angle);
            if (angle_diff < angle_diff_min) {
                angle_diff_min = angle_diff;
                closest_index_1 = j;
                closest_index_2 = (j + 1) % n;
            }
        }

        // Perform linear interpolation
        float t = (target_angle - polar_convex_theta[closest_index_1]) /
                  (polar_convex_theta[closest_index_2] - polar_convex_theta[closest_index_1]);

        float interpolated_radius = polar_convex[closest_index_1] +
                                     t * (polar_convex[closest_index_2] - polar_convex[closest_index_1]);

        evenly_polar_convex.push_back(interpolated_radius);
        evenly_polar_convex_theta.push_back(target_angle);
    }

    // Check output dimensions
    if (evenly_polar_convex.size() != num_points || evenly_polar_convex_theta.size() != num_points) {
        ROS_ERROR("Output vector dimensions do not match the expected number of points.");
        return false;
    }

    return true;
}

double cal_cross_product(const Point& p1, const Point& p2, const Point& p3)
{
    Point v1 = std::make_pair(p2.first - p1.first, p2.second - p1.second);
    Point v2 = std::make_pair(p3.first - p1.first, p3.second - p1.second);
    return v1.first * v2.second - v1.second * v2.first;
}

bool Graham_Andrew_Scan(Result& res,const Points& points)
{
    int n = points.size();


    std::vector<int> sort_index(n);
    for (int i = 0; i < n; i++)
    {
        sort_index[i] = i;
    }

    if (n < 3)
    {
        res = std::make_pair(points, sort_index);
        return true;
    }

    auto cmp = [&](int x, int y)
    {
        return points[x] < points[y];
    };
    std::sort(sort_index.begin(), sort_index.end(), cmp);

    int count = 0;
    int last_count = 1;
    Points res_points;
    std::vector<int> indexs;
    for (int i = 0; i < 2 * n; i++)
    {
        if (i == n)
        {
            last_count = count;
        }
        int index = i < n ? sort_index[i] : sort_index[2 * n - 1 - i];
        const Point& pt = points[index];

        while (count > last_count && cal_cross_product(res_points[count - 2], res_points[count - 1], pt) <= 0)
        {
            res_points.pop_back();
            indexs.pop_back();
            count--;
        }
        res_points.push_back(pt);
        indexs.push_back(index);
        count++;
    }
    count--;
    if (count < 3)
    {
        return false;
    }

    res = std::make_pair(res_points, indexs);
    return true;
}

// 圆心角角度差为依据稀疏化
// 
void sparse_scan(Result2& res,const Points& scans_xy, double drad)
{
    size_t scans_size = scans_xy.size();
    int now_index = 1;
    Points points;
    std::vector<double> thetas;
    double prev_theta = std::atan2(scans_xy[0].second,scans_xy[0].first);
    double now_theta = 0.;
    double dtheta = 0.;
    points.push_back(scans_xy[0]);
    thetas.push_back(std::atan2(scans_xy[0].second, scans_xy[0].first));
    while (now_index < scans_size - 1)
    {
        const auto& p = scans_xy[now_index];
        now_theta = std::atan2(p.second,p.first);
        dtheta = NormalizeAngle(now_theta - prev_theta);

        if (std::fabs(dtheta) > drad + 1e-6)
        {
            points.push_back(p);
            thetas.push_back(std::atan2(p.second, p.first));
            prev_theta = now_theta;
        }

        ++now_index;
    }
    points.push_back(scans_xy[0]);
    thetas.push_back(std::atan2(scans_xy[0].second, scans_xy[0].first));

    res = std::make_pair(points, thetas);
}

bool is_in_convex(const Point& p,const Points& convex,bool is_clock_wise)
{
    double cross = 0;
    size_t points_num = convex.size();
    if(points_num < 3)
        return false;
    for(int i = 1;i < points_num;++i)
    {
        cross = convex[i-1].first*convex[i].second - convex[i-1].second*convex[i].first 
        + (convex[i].first - convex[i-1].first)*p.second 
        + (convex[i-1].second - convex[i].second)*p.first;
        if((cross < 0 && !is_clock_wise)||(cross > 0 && is_clock_wise))
            return false;
    }
    return true;
}

/*
好的，我来更简洁地解释这个 `Q_equitable_distribution` 函数。
假设你负责在派对上分配100(n)块巧克力给三个小组，每个小组的人数比例分别是10%，30%和60%(arr)。
由于巧克力无法分割，你不能直接按比例分配（即10块、30块和60块），因为这样总数会超过100块。
这个函数首先尝试按比例分配，但由于只能分配整数块巧克力，它可能会先分配9块、30块和60块。
然后，它会计算哪个小组相对于它们应得的比例得到的巧克力最少，并将剩下的1块巧克力分配给这个小组，以确保分配尽可能公平(res)。
*/
void Q_equitable_distribution(std::vector<int>& res,std::vector<double>& arr,int n)
{
    std::vector<double> Q(arr.size(),0.);
    std::vector<int> arg(arr.size(),0);
    double sum = 0.;
    int m = n;
    std::vector<int> sort_Q_index(arr.size(),0);


    for (int i = 0; i < arr.size(); i++)
    {
        sum += arr[i];
        sort_Q_index[i] = i;
    }

    for(int i = 0;i < arr.size();++i)
    {
        arg[i] = floor(n*arr[i]/sum);
        Q[i] = arg[i]*(arg[i]+1)/(arr[i]*arr[i]);
        m -= arg[i];
    }

    auto cmp = [&](int x, int y)
    {
        return Q[x] < Q[y];
    };

    std::sort(sort_Q_index.begin(), sort_Q_index.end(), cmp);


    for(int i = 0;i < m;++i)
    {
        arg[sort_Q_index[i]] += 1;

    }

    res = arg;
}


bool galaxy_xyin_360out(Result3& res,const Points& scans_xy,int max_vertex_num,double origin_x = 0.0, double origin_y = 0.0, double radius = 10.0,bool if_evenly_convex = false)
{
    int lidar_num = scans_xy.size();
    // double drad = 2 * M_PI / lidar_num;
    double drad = 2 * M_PI / max_vertex_num;    
    // plt.scatter([points[i].first for i in range(len(points))],[points[i].second for i in range(len(points))],c='y')

    Points flip_data;
    double safe_radius = radius;
    for (const auto & p : scans_xy)
    {
        double dx = p.first - origin_x;
        double dy = p.second - origin_y;
        double norm2 = std::hypot(p.first, p.second);
        if (norm2 < safe_radius)
        {
            safe_radius = norm2;
        }
        if (norm2 < 1e-6)
        {
            continue;
        }
        flip_data.emplace_back(dx + 2 * (radius - norm2) * dx / norm2, dy + 2 * (radius - norm2) * dy / norm2);
    }
    // convex hull
    Result convex_points_index_pair;
    

    // flip点云的凸包，反射后不一定为凸，所以下面还有再计算一次凸包
    if (!Graham_Andrew_Scan(convex_points_index_pair,flip_data)||convex_points_index_pair.second.empty())
    {
        std::cout << "first Graham_Andrew_Scan" << std::endl;
        ROS_INFO("first Graham_Andrew_Scan");
        return false;
    }


    std::vector<int> &vertex_indexs = convex_points_index_pair.second;
    vertex_indexs.pop_back(); // 最后一个点是重复的

    Points vertexs;

    for (auto i : vertex_indexs)
    {
        vertexs.emplace_back(scans_xy[i]); // 反射后点云的凸包
    }
    // plt.plot([vertexs[i].first for i in range(len(vertexs))],[vertexs[i].second for i in range(len(vertexs))],c='g')
    double interior_x = origin_x;
    double interior_y = origin_y;


    // 要求逆时针，计算反射后点云凸包
    if (!Graham_Andrew_Scan(convex_points_index_pair,vertexs)||convex_points_index_pair.second.empty())
    {
        std::cout << "second Graham_Andrew_Scan" << std::endl;
        ROS_INFO("second Graham_Andrew_Scan");
        return false;
    }
    std::vector<int> & indexs = convex_points_index_pair.second;
    Points is_feasible_convex;
    for(int i = 0;i < indexs.size();++i)
    {
        is_feasible_convex.emplace_back(vertexs[indexs[i]]);
    }
    if(!is_in_convex(Point(0.,0.),is_feasible_convex,false))
    {
        std::cout << "first is_in_convex" << std::endl;
        ROS_INFO("first is_in_convex");
        return false;
    }
    indexs.pop_back();



    // 顶点的约束
    std::vector<std::tuple <double, double , double>> constraints;
    for (size_t j = 0; j < indexs.size(); ++j)
    {
        size_t jplus1 = (j + 1) % indexs.size();
        // 星型多边形的一条边  
        double rayV_x = vertexs[indexs[jplus1]].first - vertexs[indexs[j]].first;
        double rayV_y = vertexs[indexs[jplus1]].second - vertexs[indexs[j]].second;
        // 顺时针转90度
        double normalj_x = rayV_y; // point to outside
        double normalj_y = -rayV_x;
        double norm = std::hypot(normalj_x, normalj_y);
        normalj_x /= norm;
        normalj_y /= norm;
        size_t ij = indexs[j];

        while (ij != indexs[jplus1])
        {
            // 多边形内部三角形(以多边形边作为底边，以原点和顶点连线作为其他边)腰边在高方向的投影
            double c = (vertexs[ij].first - interior_x) * normalj_x + (vertexs[ij].second - interior_y) * normalj_y + 1e-6;
            constraints.emplace_back(normalj_x, normalj_y, c);
            ij = (ij + 1) % vertexs.size();
        }
    }
    Points dual_points;
    for (const std::tuple <double, double , double>& c : constraints)
    {        
        dual_points.emplace_back(std::get<0>(c) / std::get<2>(c), std::get<1>(c) / std::get<2>(c));
    }

    // 顺时针
    if (!Graham_Andrew_Scan(convex_points_index_pair,dual_points)||convex_points_index_pair.first.empty())
    {
        std::cout << "3th Graham_Andrew_Scan" << std::endl;
        ROS_INFO("3th Graham_Andrew_Scan");
        return false;
    }

    Points& dual_verterx = convex_points_index_pair.first;
    dual_verterx.pop_back();
    std::reverse(dual_verterx.begin(), dual_verterx.end());
    Points final_vertex;

    // 输出顺时针
    for (size_t i = 0; i < dual_verterx.size(); ++i)
    {
        size_t iplus1 = (i + 1) % dual_verterx.size();
        double rayi_x = dual_verterx[iplus1].first - dual_verterx[i].first;
        double rayi_y = dual_verterx[iplus1].second - dual_verterx[i].second;
        double c = rayi_y * dual_verterx[i].first - rayi_x * dual_verterx[i].second;
        final_vertex.emplace_back(interior_x + rayi_y / c, interior_y - rayi_x / c);
    }
    final_vertex.emplace_back(final_vertex[0]);

    Result2 points_thetas_pair;
    // 不需要稀疏
    if(final_vertex.size() <= max_vertex_num + 1)
    {
        for(auto& p:final_vertex)
        {
            points_thetas_pair.first.push_back(p);
            points_thetas_pair.second.push_back(std::atan2(p.second, p.first));
        }
    }
    else
        // 转为与scans数量相同的极坐标形式
        sparse_scan(points_thetas_pair,final_vertex, drad);


    Points& convex = points_thetas_pair.first;
    std::vector<double>& thetas = points_thetas_pair.second;
    
    int res_num = max_vertex_num - convex.size() + 1;   
    // 分配角度范围
    // thetas.push_back(thetas[0]);
    // convex.push_back(convex[0]);
    if(!is_in_convex(Point(0.,0.),convex,true))
    {
        std::cout << "2th is_in_convex" << std::endl;
        ROS_INFO("2th is_in_convex");
        return false;
    }
    

    if(res_num < 0)
        std::cout<<res_num<<" size "<< convex.size() << "," << thetas.size() << std::endl;
    int s = thetas.size();
    if(s - 1 < 0)
        std::cout<<convex.size() <<" "<< thetas.size() <<"dtheta size neg"<< std::endl;

    // std::cout<<"("<<convex[0].first << "," << convex[0].second <<")  " 
    //     <<"("<<convex[s-2].first << "," << convex[s-2].second <<")  " 
    //     <<"("<<convex[s-1].first << "," << convex[s-1].second <<")  " 
    //     << std::endl;

    // max_vertex_num >= convex.size()

    
    std::vector<int> nums;
    // 初始化向量 dthetas: 首先，代码创建了一个双精度浮点数向量 dthetas，其大小为 thetas.size() - 1，并且所有元素都初始化为 0.0。
    // 这个向量用来存储连续两个角度之间的差值。
    std::vector<double> dthetas(thetas.size() - 1,0.);
    
    // 计算角度差值: 接着，代码遍历 dthetas 向量，并计算 thetas 向量中每两个连续角度的差值。这里使用的 NormalizeAngleTo2Pi 函数将角度差值标准化到 [0, 2π] 的范围内。标准化是为了确保角度差值是正数，并且在一个完整的圆周范围内。
    for(int i = 0;i < dthetas.size(); ++i)
    {
        dthetas[i] = NormalizeAngleTo2Pi(thetas[i] - thetas[i + 1]);
    }

    // 调用 Q_equitable_distribution 函数: 
    // 最后，代码调用了 Q_equitable_distribution 函数，将 nums 向量、dthetas 向量以及 res_num 作为参数传递。
    // Q_equitable_distribution 函数的作用是根据 dthetas 中的角度差值分配 res_num 数量的资源。
    // 这个分配过程是为了确保每个角度间隔获得的资源数量尽可能公平。
    // 例如，如果 dthetas 向量中的角度差值都是 0.1，那么 Q_equitable_distribution 函数将会将 res_num 个资源分配给每个角度间隔，这样每个角度间隔都会获得相同数量的资源。
    Q_equitable_distribution(nums,dthetas,res_num);



    // for(int i = 0;i < dthetas.size(); ++i)
    // {
    //     std::cout<<"("<<dthetas[i]<<","<<nums[i]<<") ";
    // }
    // std::cout<<std::endl;


    std::vector<float> polar_convex;
    std::vector<float> polar_convex_theta;
    std::vector<float> evenly_polar_convex;
    std::vector<float> evenly_polar_convex_theta;

    // 將角度變爲0-2pi，並找到角度最小且大於0角的索引
    int first_pos = -1;
    float min_pos_theta = 2*M_PI;
    // 转换为极坐标：遍历 convex 向量中的每个点（除了最后一个点）。对于每个点 p，代码计算它与下一个点之间的距离 dist 和角度 t
    // 。然后，代码遍历 nums 向量中的每个元素 j，并计算点 p 和点 p + j 之间的距离。这里的点 p + j 是点 p 沿着方向 t 移动 dist * j / nums[i] 距离后的点。
    for (size_t i = 0; i < convex.size() - 1; ++i)
    {
        const auto &p = convex[i];
        double dist = std::hypot(p.first - convex[i + 1].first, p.second - convex[i + 1].second);
        double t = std::atan2(convex[i + 1].second - p.second, convex[i + 1].first - p.first);
        double dir_x = std::cos(t);
        double dir_y = std::sin(t);

        // 线性插值：对于每一对相邻点，代码在它们之间进行线性插值以生成中间点。插值的数量由 nums[i] 确定，这可能是之前某个步骤中计算出的每个间隔应有的点数。对于每个新生成的点，计算其在极坐标下的位置（距离和角度）并添加到 polar_convex 和 polar_convex_theta。
        for (size_t j = 0; j <= nums[i]; ++j)
        {
            double x = p.first + j * dir_x / (nums[i]+1) * dist;
            double y = p.second + j * dir_y / (nums[i]+1) * dist;

            polar_convex.push_back(std::hypot(x, y));
            polar_convex_theta.push_back(std::atan2(y, x));
            if(polar_convex_theta.back() < 0)
                polar_convex_theta.back() += 2*M_PI;

            // 找到角度最小的点：在添加每个新点的同时，代码检查其角度是否小于当前记录的最小角度 min_pos_theta。如果是，则更新 min_pos_theta 并记录该点的位置 first_pos。
            if(min_pos_theta > polar_convex_theta.back())
            {
                first_pos = polar_convex_theta.size();
                min_pos_theta = polar_convex_theta.back();
            }
            
        }
    }

    // 检查并更新 first_pos：
    // 如果 first_pos 仍然是初始值 -1，则表示在之前的步骤中没有找到具有最小正角度的点。
    // 代码遍历 polar_convex_theta 向量，查找具有最小角度的点，并更新 first_pos 为该点的索引。
    if(first_pos == -1)
    {
        for(int i = 0;i < polar_convex_theta.size();++i)
        {
  
            if(min_pos_theta > polar_convex_theta[i])
            {
                first_pos = i;
                min_pos_theta = polar_convex_theta[i];
            }
            
        }
    }

    //     旋转向量：
    // 使用 std::rotate 函数对 polar_convex_theta 和 polar_convex 向量进行循环左移操作。
    // 左移的目的是使得具有最小角度的点成为向量的第一个元素。这种操作可以使多边形的起点对应于极坐标中的最小角度点。
    std::rotate(polar_convex_theta.begin(),polar_convex_theta.begin() + first_pos,polar_convex_theta.end());// 循環左移
    std::rotate(polar_convex.begin(),polar_convex.begin() + first_pos,polar_convex.end());// 循環左移


    // 反转向量：
    // 对 polar_convex 和 polar_convex_theta 向量进行反转。反转的目的可能是为了调整点的顺序，使其符合特定的顺序要求，例如顺时针或逆时针。
    // std::reverse(convex.begin(), convex.end()); // 不轉成逆時針 因爲action中都以順時針來計算
    std::reverse(polar_convex.begin(), polar_convex.end());
    std::reverse(polar_convex_theta.begin(), polar_convex_theta.end());
    

    // 构建多边形消息：
    // 代码构建了一个 geometry_msgs::Polygon 类型的对象 polygon。
    // 遍历原始的 convex 向量中的每个顶点，并将其添加到 polygon 的点列表中。这里使用了 geometry_msgs::Point32 类型来存储每个顶点的坐标。
    // 这个过程实际上是在将凸多边形的顶点坐标转换为可以在消息传递系统中使用的格式，这在机器人操作系统（如ROS）中很常见。
    geometry_msgs::Polygon polygon;
    geometry_msgs::Point32 p;
    for(const auto& vert:convex)
    {
        p.x = vert.first;
        p.y = vert.second;
        polygon.points.emplace_back(p);
    }

    if (if_evenly_convex && evenlyDistributePoints(polar_convex, polar_convex_theta,
                               evenly_polar_convex, evenly_polar_convex_theta,
                               max_vertex_num)) {
        // polygon：它由一系列点组成，这些点定义了多边形的顶点。在这段代码中，polygon 的顶点来自 convex 向量，其中每个顶点由一个 geometry_msgs::Point32 类型的对象表示。Point32 对象包含三个浮点数字段 x, y, z，在这种情况下只使用 x 和 y 来表示二维空间中的点。
        // polar_convex、polar_convex_theta：极坐标下的距离、角度。相比于convex，polar_convex已经根据max_vertex_num计算重新Q分配后的点
        res = std::tie(polygon,evenly_polar_convex,evenly_polar_convex_theta);  
    } else {
        // polygon：它由一系列点组成，这些点定义了多边形的顶点。在这段代码中，polygon 的顶点来自 convex 向量，其中每个顶点由一个 geometry_msgs::Point32 类型的对象表示。Point32 对象包含三个浮点数字段 x, y, z，在这种情况下只使用 x 和 y 来表示二维空间中的点。
        // polar_convex、polar_convex_theta：极坐标下的距离、角度。相比于convex，polar_convex已经根据max_vertex_num计算重新Q分配后的点
        res = std::tie(polygon,polar_convex,polar_convex_theta);
    }
    return true;
}