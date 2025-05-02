#include <deque>
#include <atomic>
#include <thread>
#include <numeric>
#include <Eigen/Core>

#define GLIM_ROS2

#include <boost/format.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/concurrent_vector.hpp>

#ifdef GLIM_ROS2
#include <glim/util/extension_module_ros2.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <GeographicLib/UTMUPS.hpp>

using ExtensionModuleBase = glim::ExtensionModuleROS2;
using NavSatFix = sensor_msgs::msg::NavSatFix;
using NavSatFixConstPtr = sensor_msgs::msg::NavSatFix::ConstSharedPtr;

using namespace GeographicLib;

template <typename Stamp>
double to_sec(const Stamp& stamp) {
  return stamp.sec + stamp.nanosec / 1e9;
}
#else
#include <glim/util/extension_module_ros.hpp>
#include <sensor_msgs/NavSatFix.h>
using ExtensionModuleBase = glim::ExtensionModuleROS;
#endif

#include <spdlog/spdlog.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PoseTranslationPrior.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <glim/util/logging.hpp>
#include <glim/util/convert_to_string.hpp>
#include <glim_ext/util/config_ext.hpp>

namespace glim {

using gtsam::symbol_shorthand::X;

struct GNSSData {
  double timestamp;
  Eigen::Vector3d position;
  Eigen::Matrix3d covariance;
};

class GNSSGlobal : public ExtensionModuleBase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GNSSGlobal() : logger(create_module_logger("gnss_global")) {
    logger->info("initializing GNSS global constraints");
    const std::string config_path = glim::GlobalConfigExt::get_config_path("config_gnss_global");
    logger->info("gnss_global_config_path={}", config_path);

    glim::Config config(config_path);
    gnss_topic = config.param<std::string>("gnss", "gnss_topic", "/fix");
    prior_inf_scale = config.param<Eigen::Vector3d>("gnss", "prior_inf_scale", Eigen::Vector3d(1e3, 1e3, 0.0));
    min_baseline = config.param<double>("gnss", "min_baseline", 5.0);

    transformation_initialized = false;
    T_world_utm.setIdentity();

    kill_switch = false;
    thread = std::thread([this] { backend_task(); });

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    GlobalMappingCallbacks::on_insert_submap.add(std::bind(&GNSSGlobal::on_insert_submap, this, _1));
    GlobalMappingCallbacks::on_smoother_update.add(std::bind(&GNSSGlobal::on_smoother_update, this, _1, _2, _3));
  }
  ~GNSSGlobal() {
    kill_switch = true;
    thread.join();
  }

  virtual std::vector<GenericTopicSubscription::Ptr> create_subscriptions() override {
    const auto sub = std::make_shared<TopicSubscription<NavSatFix>>(gnss_topic, [this](const NavSatFixConstPtr msg) { gnss_callback(msg); });
    return {sub};
  }

  Eigen::Vector3d latlon_to_utm(double lat, double lon, double alt) {
    double northing, easting;
    int zone;
    bool northp;
    UTMUPS::Forward(lat, lon, zone, northp, easting, northing);
    return Eigen::Vector3d(easting, northing, alt);
  }

  void gnss_callback(const NavSatFixConstPtr& msg) {
    GNSSData data;
    //logger->info("GNSS data received: lat={}, lon={}, alt={}", msg->latitude, msg->longitude, msg->altitude);
    data.timestamp = to_sec(msg->header.stamp);
    data.position = latlon_to_utm(msg->latitude, msg->longitude, msg->altitude);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        data.covariance(i, j) = msg->position_covariance[i * 3 + j];

    // Apply scaling heuristic to convert covariance from geodetic to UTM frame
    // double lat_scale = 111000.0;
    // double lon_scale = 111000.0 * std::cos(msg->latitude * M_PI / 180.0);

    // Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    // S(0, 0) = lon_scale;
    // S(1, 1) = lat_scale;
    // S(2, 2) = 1.0;  // altitude assumed already in meters

    // Eigen::Matrix3d cov_geo;
    // for (int i = 0; i < 3; ++i)
    //   for (int j = 0; j < 3; ++j)
    //     cov_geo(i, j) = msg->position_covariance[i * 3 + j];

    // data.covariance = S * cov_geo * S.transpose();
    
    input_gnss_queue.push_back(data);
  }

  void on_insert_submap(const SubMap::ConstPtr& submap) { input_submap_queue.push_back(submap); }

  void on_smoother_update(gtsam_points::ISAM2Ext& isam2, gtsam::NonlinearFactorGraph& new_factors, gtsam::Values& new_values) {
    const auto factors = output_factors.get_all_and_clear();
    if (!factors.empty()) {
      logger->debug("insert {} GNSS prior factors", factors.size());
      new_factors.add(factors);
    }
  }

  void backend_task() {
    std::deque<GNSSData> gnss_queue;
    std::deque<SubMap::ConstPtr> submap_queue;

    while (!kill_switch) {
      auto new_gnss = input_gnss_queue.get_all_and_clear();
      gnss_queue.insert(gnss_queue.end(), new_gnss.begin(), new_gnss.end());

      auto new_submaps = input_submap_queue.get_all_and_clear();
      if (new_submaps.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }
      submap_queue.insert(submap_queue.end(), new_submaps.begin(), new_submaps.end());

      while (!gnss_queue.empty() && !submap_queue.empty() && submap_queue.front()->frames.front()->stamp < gnss_queue.front().timestamp) {
        submap_queue.pop_front();
      }

      while (!gnss_queue.empty() && !submap_queue.empty()) {
        const auto& submap = submap_queue.front();
        double submap_stamp = submap->frames[submap->frames.size() / 2]->stamp;

        auto right = std::lower_bound(gnss_queue.begin(), gnss_queue.end(), submap_stamp, [](const GNSSData& d, double t) { return d.timestamp < t; });
        if (right == gnss_queue.end() || right == gnss_queue.begin()) break;

        auto left = right - 1;
        double t1 = left->timestamp, t2 = right->timestamp;
        double p = (submap_stamp - t1) / (t2 - t1);

        Eigen::Vector3d interp_pos = (1 - p) * left->position + p * right->position;
        Eigen::Matrix3d interp_cov = (1 - p) * left->covariance + p * right->covariance;

        submaps.push_back(submap);
        submap_coords.push_back({submap_stamp, interp_pos, interp_cov});

        submap_queue.pop_front();
        gnss_queue.erase(gnss_queue.begin(), left);
      }

      if (!transformation_initialized && !submaps.empty() &&
          (submaps.front()->T_world_origin.inverse() * submaps.back()->T_world_origin).translation().norm() > min_baseline) {
        Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
        Eigen::Vector3d mean_gnss = Eigen::Vector3d::Zero();
        for (int i = 0; i < submaps.size(); ++i) {
          mean_est += submaps[i]->T_world_origin.translation();
          mean_gnss += submap_coords[i].position;
        }
        mean_est /= submaps.size();
        mean_gnss /= submaps.size();

        Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
        for (int i = 0; i < submaps.size(); ++i) {
          Eigen::Vector3d d_est = submaps[i]->T_world_origin.translation() - mean_est;
          Eigen::Vector3d d_gnss = submap_coords[i].position - mean_gnss;
          cov += d_gnss * d_est.transpose();
        }
        cov /= submaps.size();

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov.block<2, 2>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d U = svd.matrixU();
        Eigen::Matrix2d V = svd.matrixV();
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity();
        if ((U * V.transpose()).determinant() < 0) S(1, 1) = -1;

        Eigen::Isometry3d T_utm_world = Eigen::Isometry3d::Identity();
        T_utm_world.linear().block<2, 2>(0, 0) = U * S * V.transpose();
        T_utm_world.translation() = mean_gnss - T_utm_world.linear() * mean_est;

        T_world_utm = T_utm_world.inverse();
        transformation_initialized = true;
      }

      if (transformation_initialized && !submaps.empty()) {
        const auto& submap = submaps.back();
        const auto& gnss_data = submap_coords.back();

        Eigen::Vector3d xyz = T_world_utm * gnss_data.position;
        Eigen::Matrix3d cov = gnss_data.covariance;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
        Eigen::Vector3d eigvals = solver.eigenvalues().cwiseMax(1e-3);
        Eigen::Matrix3d reg_cov = solver.eigenvectors() * eigvals.asDiagonal() * solver.eigenvectors().transpose();
        Eigen::Matrix3d info = reg_cov.inverse(); // using regularized covariance from GPS signal
        //Eigen::Matrix3d info = prior_inf_scale.asDiagonal();  // using fixed prior information scale
        // logger->info("integrating GNSS");
        // std::stringstream ss;
        // ss << info;
        // logger->info("GNSS information matrix:\n{}", ss.str());

        

        auto model = gtsam::noiseModel::Gaussian::Information(info);
        auto factor = boost::make_shared<gtsam::PoseTranslationPrior<gtsam::Pose3>>(X(submap->id), xyz, model);
        logger->info("GNSS prior factor added");
        output_factors.push_back(factor);

      }
    }
  }

private:
  std::atomic_bool kill_switch;
  std::thread thread;

  ConcurrentVector<GNSSData> input_gnss_queue;
  ConcurrentVector<SubMap::ConstPtr> input_submap_queue;
  ConcurrentVector<gtsam::NonlinearFactor::shared_ptr> output_factors;

  std::vector<SubMap::ConstPtr> submaps;
  std::vector<GNSSData> submap_coords;

  std::string gnss_topic;
  Eigen::Vector3d prior_inf_scale;
  double min_baseline;

  bool transformation_initialized;
  Eigen::Isometry3d T_world_utm;

  std::shared_ptr<spdlog::logger> logger;
};

}  // namespace glim

extern "C" glim::ExtensionModule* create_extension_module() {
  return new glim::GNSSGlobal();
}
