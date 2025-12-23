#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_control_mode.hpp>
#include <px4_msgs/msg/vehicle_attitude.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <qjwdds/msg/image_deviation.hpp>
#include <stdint.h>
#include <mutex>
#include <cmath>
#include <chrono>
#include <iostream>
#include <atomic>
#include <limits>

using namespace std::chrono;
using namespace std::chrono_literals;
using namespace px4_msgs::msg;

// 角度归一化到[-π, π]
inline double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

class VisionControl : public rclcpp::Node
{
public:
    VisionControl() : Node("sh_vision_control"),
                      offboard_setpoint_counter_(0),
                      current_state_(State::TAKEOFF),
                      has_new_deviation_(false),
                      has_attitude_data_(false),
                      has_odometry_(false),
                      has_arm_position_(false),
                      has_valid_target_(false),
                      current_yaw_(0.0),
                      desired_yaw_(0.0),
                      takeoff_yaw_(0.0),
                      last_angle_x_(0.0),
                      last_angle_y_(0.0),
                      last_valid_angle_x_(0.0),
                      last_valid_angle_y_(0.0),
                      target_loss_count_(0),
                      arm_position_x_(0.0),
                      arm_position_y_(0.0),
                      base_altitude_(0.0),
                      hover_position_x_(0.0),
                      hover_position_y_(0.0),
                      hover_position_z_(0.0),
                      target_velocity_x_(0.0),
                      target_velocity_y_(0.0),
                      target_velocity_z_(0.0)
    {
        // 参数声明
        declare_parameter("takeoff_relative_altitude", 10.0);    // 相对解锁高度的起飞高度(米)
        declare_parameter("takeoff_timeout", 30.0);             // 起飞超时时间(秒)
        declare_parameter("proportional_gain", 4.0);            // 水平比例增益
        declare_parameter("vertical_gain", 10.0);                // 垂直
        declare_parameter("yaw_gain", 1.0);                     // 偏航
        declare_parameter("max_speed", 5.0);                    // 最大水平速度(m/s)
        declare_parameter("max_vertical_speed", 5.0);           // 最大垂直速度(m/s)
        declare_parameter("max_yaw_rate", 0.5);                 // 最大偏航角速度(rad/s)
        declare_parameter("deviation_topic", "/camera/image_deviation"); 
        declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude");
        declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry");
        declare_parameter("min_relative_altitude", 0.5);        // 最低相对高度(米)
        declare_parameter("max_relative_altitude", 50.0);       // 最高相对高度(米)
        declare_parameter("target_loss_timeout", 0.5);          // 目标丢失超时时间(秒)
        declare_parameter("target_loss_max_count", 25);         // 最大连续丢失次数
        
        takeoff_relative_altitude_ = get_parameter("takeoff_relative_altitude").as_double();
        takeoff_timeout_ = get_parameter("takeoff_timeout").as_double();
        proportional_gain_ = get_parameter("proportional_gain").as_double();
        vertical_gain_ = get_parameter("vertical_gain").as_double();
        yaw_gain_ = get_parameter("yaw_gain").as_double();
        max_speed_ = get_parameter("max_speed").as_double();
        max_vertical_speed_ = get_parameter("max_vertical_speed").as_double();
        max_yaw_rate_ = get_parameter("max_yaw_rate").as_double();
        min_relative_altitude_ = get_parameter("min_relative_altitude").as_double();
        max_relative_altitude_ = get_parameter("max_relative_altitude").as_double();
        deviation_topic_ = get_parameter("deviation_topic").as_string();
        attitude_topic_ = get_parameter("attitude_topic").as_string();
        odometry_topic_ = get_parameter("odometry_topic").as_string();
        target_loss_timeout_ = get_parameter("target_loss_timeout").as_double();
        target_loss_max_count_ = get_parameter("target_loss_max_count").as_int();

        RCLCPP_INFO(this->get_logger(), "Starting Vision Control");
        RCLCPP_INFO(this->get_logger(), "Takeoff altitude: %.1f m",
                   takeoff_relative_altitude_);
        RCLCPP_INFO(this->get_logger(), "Target loss tolerance: timeout=%.2f s, max count=%d",
                   target_loss_timeout_, target_loss_max_count_);

        takeoff_start_time_ = this->now();

        // 创建发布器
        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 50);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 50);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(
            "/fmu/in/vehicle_command", 50);
        
        // 创建订阅器
        image_deviation_sub_ = this->create_subscription<qjwdds::msg::ImageDeviation>(
            deviation_topic_, rclcpp::SensorDataQoS(),
            [this](const qjwdds::msg::ImageDeviation::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(deviation_mutex_);
                last_angle_x_ = msg->angle_x;    
                last_angle_y_ = msg->angle_y;    
                has_new_deviation_ = true;

                if (!std::isnan(last_angle_x_) && !std::isnan(last_angle_y_)) {
                    last_valid_angle_x_ = last_angle_x_;
                    last_valid_angle_y_ = last_angle_y_;

                    if (!has_valid_target_) {
                        last_valid_target_time_ = this->now();
                        has_valid_target_ = true;
                        RCLCPP_INFO(this->get_logger(), "First valid target detected");
                    } else {
                        last_valid_target_time_ = this->now();
                    }
                    
                    target_loss_count_ = 0;
                }
            });
        
        vehicle_attitude_sub_ = this->create_subscription<VehicleAttitude>(
            attitude_topic_, rclcpp::SensorDataQoS(),
            [this](const VehicleAttitude::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(attitude_mutex_);
                current_attitude_q_ = msg->q;
                current_yaw_ = getYawFromQuaternion(msg->q);
                has_attitude_data_ = true;
            });
        
        vehicle_odometry_sub_ = this->create_subscription<VehicleOdometry>(
            odometry_topic_, rclcpp::SensorDataQoS(),
            [this](const VehicleOdometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(odometry_mutex_);
                current_position_x_ = msg->position[0];
                current_position_y_ = msg->position[1];
                current_position_z_ = msg->position[2];
                has_odometry_ = true;
            });
        
        // 主控制定时器(50Hz = 20ms)
        timer_ = this->create_wall_timer(20ms, std::bind(&VisionControl::timer_callback, this));
    }

private:
    enum class State {
        TAKEOFF,
        HOVER, 
        GUIDANCE
    };
    
    // 定时器与时间
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time takeoff_start_time_;  //起飞开始时间  
    rclcpp::Time last_valid_target_time_;  // 最后一次收到有效目标的时间
    
    // 发布器 订阅器
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    
    rclcpp::Subscription<qjwdds::msg::ImageDeviation>::SharedPtr image_deviation_sub_;
    rclcpp::Subscription<VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_sub_;
    
    // 状态变量
    uint64_t offboard_setpoint_counter_;  // 切换Offboard模式的计数  
    State current_state_;                 // 状态机状态                 
    bool has_new_deviation_;              // 是否有新的目标偏差数据              
    bool has_attitude_data_;              // 是否获取到姿态数据              
    bool has_odometry_;                   // 是否获取到里程计数据                   
    bool has_arm_position_;               // 是否记录了解锁位置               
    bool has_valid_target_;               // 是否存在过有效目标
    
    // 姿态数据
    std::mutex attitude_mutex_;
    std::array<float, 4> current_attitude_q_;  // 当前姿态四元数  
    double current_yaw_;                       // 当前偏航角(rad)
    double desired_yaw_;                       // 期望偏航角(rad)
    double takeoff_yaw_;                       // 起飞时的初始偏航角(rad)                       
    
    // 位置数据
    std::mutex odometry_mutex_;
    double current_position_x_;  // 当前X位置(NED)
    double current_position_y_;  // 当前Y位置(NED)
    double current_position_z_;  // 当前Z位置(NED)  
    
    // 视觉偏差数据
    std::mutex deviation_mutex_;
    double last_angle_x_;  // 目标相对航向的水平角度(度，右正左负)
    double last_angle_y_;  // 目标相对俯仰的垂直角度(度，下正上负)  
    double last_valid_angle_x_;  // 上一次有效的水平角度
    double last_valid_angle_y_;  // 上一次有效的垂直角度
    int target_loss_count_;      // 连续丢失目标的计数
    
    // 解锁基准数据
    double arm_position_x_;  // 解锁位置X(NED)
    double arm_position_y_;  // 解锁位置Y(NED)
    double base_altitude_;   // 解锁时的Z坐标(NED)
    
    // 记录状态(悬停使用)
    double hover_position_x_;  // 悬停X位置
    double hover_position_y_;  // 悬停Y位置
    double hover_position_z_;  // 悬停Z位置
    
    // 期望速度
    double target_velocity_x_;  // 期望X速度(NED)
    double target_velocity_y_;  // 期望Y速度(NED)
    double target_velocity_z_;  // 期望Z速度(NED)  
    
    // 控制参数
    double takeoff_relative_altitude_;  // 起飞相对高度(米)
    double takeoff_timeout_;            // 起飞超时时间(秒)
    double proportional_gain_;          // 水平比例增益
    double vertical_gain_;              // 垂直比例增益
    double yaw_gain_;                   // 偏航控制增益
    double max_speed_;                  // 最大水平速度(m/s)
    double max_vertical_speed_;         // 最大垂直速度(m/s)
    double max_yaw_rate_;               // 最大偏航角速度(rad/s)
    double min_relative_altitude_;      // 最低相对高度(米)
    double max_relative_altitude_;      // 最高相对高度(米)
    std::string deviation_topic_;       // 偏差话题名
    std::string attitude_topic_;        // 姿态话题名
    std::string odometry_topic_;        // 里程计话题名        
    double target_loss_timeout_;         // 目标丢失超时时间(秒)
    int target_loss_max_count_;          // 最大连续丢失次数
    
    /**
     * 从四元数计算偏航角(yaw)
     * @param q 四元数 [w, x, y, z]
     * @return 偏航角(rad，范围[-π, π])
     */
    double getYawFromQuaternion(const std::array<float, 4>& q) {
        double siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
        double cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
        return std::atan2(siny_cosp, cosy_cosp);
    }

    /**
     * 机体坐标系到地面坐标系(NED)的向量转换
     * @param q 姿态四元数
     * @param body_vec 机体坐标系向量(x:前, y:右, z:下)
     * @return 地面坐标系向量(NED)
     */
    std::array<double, 3> bodyToGround(const std::array<float, 4>& q, const std::array<double, 3>& body_vec) {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        double R[3][3] = {
            {1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y},
            {2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x},
            {2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y}
        };
        return {
            R[0][0] * body_vec[0] + R[0][1] * body_vec[1] + R[0][2] * body_vec[2],
            R[1][0] * body_vec[0] + R[1][1] * body_vec[1] + R[1][2] * body_vec[2],
            R[2][0] * body_vec[0] + R[2][1] * body_vec[1] + R[2][2] * body_vec[2]
        };
    }

    /**
     * 偏航角控制: 根据目标水平角度调整期望偏航角
     * @param angle_x 目标相对当前航向的水平角度(度，右正左负)
     */
    void compute_yaw_control(double angle_x_deg) {
        double angle_x_rad = angle_x_deg * M_PI / 180.0;

        desired_yaw_ = current_yaw_ + angle_x_rad;
        desired_yaw_ = normalize_angle(desired_yaw_);
        
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Yaw control: current=%.2f°, aim=%.2f° (angle_x=%.1f°)",
            current_yaw_ * 57.3, desired_yaw_ * 57.3, angle_x_deg);
    }

    /**
     * 比例导引法: 计算期望速度和偏航角
     * @param angle_x 目标水平角度(度)
     * @param angle_y 目标垂直角度(度)
     */
    void proportional_guidance(double angle_x_deg, double angle_y_deg) {
        double angle_x = angle_x_deg * M_PI / 180.0;
        double angle_y = angle_y_deg * M_PI / 180.0;

        double body_x = std::cos(angle_y) * std::cos(angle_x);  // 前向分量
        double body_y = std::cos(angle_y) * std::sin(angle_x);  // 右
        double body_z = std::sin(angle_y);                      // 下

        std::array<float, 4> current_q;
        {
            std::lock_guard<std::mutex> lock(attitude_mutex_);
            current_q = current_attitude_q_;
        }
        auto ground_vec = bodyToGround(current_q, {body_x, body_y, body_z});

        double desired_vel_x = ground_vec[0] * proportional_gain_;  // 北向速度
        double desired_vel_y = ground_vec[1] * proportional_gain_;  // 东
        double desired_vel_z = ground_vec[2] * vertical_gain_;      // 下(正)

        double horizontal_speed = std::hypot(desired_vel_x, desired_vel_y);
        if (horizontal_speed > max_speed_) {
            double scale = max_speed_ / horizontal_speed;
            desired_vel_x *= scale;
            desired_vel_y *= scale;
        }

        if(desired_vel_z>max_vertical_speed_){desired_vel_z=max_vertical_speed_;}
        else if (desired_vel_z<-max_vertical_speed_){desired_vel_z=-max_vertical_speed_;}

        compute_yaw_control(angle_x_deg);

        target_velocity_x_ = desired_vel_x;
        target_velocity_y_ = desired_vel_y;
        target_velocity_z_ = desired_vel_z;

        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 500,
            "Guidance: vel(%.2f, %.2f, %.2f) m/s | angle(%.2f, %.2f) deg",
            desired_vel_x, desired_vel_y, desired_vel_z, angle_x_deg, angle_y_deg);
    }

    /**
     * 高度限制
     */
    void enforce_altitude_constraints() {
        double current_relative_alt = base_altitude_ - current_position_z_;

        if (current_relative_alt < min_relative_altitude_ && target_velocity_z_ > 0) {
            target_velocity_z_ = 0.0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Below min altitude (%.1f m), stopping descent", current_relative_alt);
        }
        else if (current_relative_alt > max_relative_altitude_ && target_velocity_z_ < 0) {
            target_velocity_z_ = 0.0;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Above max altitude (%.1f m), stopping ascent", current_relative_alt);
        }
    }

    /**
     * 发布Offboard控制模式
     */
    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        
        switch (current_state_) {
            case State::TAKEOFF:
            case State::HOVER:
                msg.position = true;
                msg.velocity = false;
                break;
                
            case State::GUIDANCE:
                msg.position = false;
                msg.velocity = true;
                break;
        }
        
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg);
    }

    /**
     * 发布轨迹设定点: 包含位置/速度和姿态
     */
    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};

        switch (current_state_) {
            case State::TAKEOFF:
                msg.position[0] = arm_position_x_;
                msg.position[1] = arm_position_y_;
                msg.position[2] = base_altitude_ - takeoff_relative_altitude_;
                msg.velocity[0] = std::numeric_limits<double>::quiet_NaN();
                msg.velocity[1] = std::numeric_limits<double>::quiet_NaN();
                msg.velocity[2] = std::numeric_limits<double>::quiet_NaN();
                msg.yaw = takeoff_yaw_;
                break;
                
            case State::HOVER:
                msg.position[0] = hover_position_x_;
                msg.position[1] = hover_position_y_;
                msg.position[2] = hover_position_z_;
                msg.velocity[0] = std::numeric_limits<double>::quiet_NaN();
                msg.velocity[1] = std::numeric_limits<double>::quiet_NaN();
                msg.velocity[2] = std::numeric_limits<double>::quiet_NaN();
                msg.yaw = current_yaw_;
                break;
                
            case State::GUIDANCE:
                msg.position[0] = std::numeric_limits<double>::quiet_NaN();
                msg.position[1] = std::numeric_limits<double>::quiet_NaN();
                msg.position[2] = std::numeric_limits<double>::quiet_NaN();
                msg.velocity[0] = target_velocity_x_;
                msg.velocity[1] = target_velocity_y_;
                msg.velocity[2] = target_velocity_z_;
                msg.yaw = desired_yaw_;
                break;
        }

        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        trajectory_setpoint_publisher_->publish(msg);
    }

    /**
     * 发布无人机控制命令
     */
    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) {
        VehicleCommand msg{};
        msg.param1 = param1;
        msg.param2 = param2;
        msg.command = command;
        msg.target_system = 1;
        msg.target_component = 1;
        msg.source_system = 1;
        msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }

    /**
     * 解锁无人机并记录基准位置
     */
    void arm() {
        std::lock_guard<std::mutex> lock(odometry_mutex_);
        if (has_odometry_) {
            arm_position_x_ = current_position_x_;
            arm_position_y_ = current_position_y_;
            base_altitude_ = current_position_z_;
            has_arm_position_ = true;

            {
                std::lock_guard<std::mutex> lock(attitude_mutex_);
                if (has_attitude_data_) {
                    takeoff_yaw_ = current_yaw_;
                    desired_yaw_ = takeoff_yaw_;
                }
            }

            RCLCPP_INFO(this->get_logger(), "Armed at (%.2f, %.2f), base Z=%.2f (NED)",
                      arm_position_x_, arm_position_y_, base_altitude_);
        } else {
            RCLCPP_ERROR(this->get_logger(), "No odometry data for arming!");
        }

        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
        RCLCPP_INFO(this->get_logger(), "Arm command sent");
    }

    /**
     * 定时器回调: 主控制(50Hz)
     */
    void timer_callback() {
        if (offboard_setpoint_counter_ < 100) {
            publish_offboard_control_mode();
            publish_trajectory_setpoint();
            offboard_setpoint_counter_++;
            
            if (offboard_setpoint_counter_ == 50) {
                publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
                RCLCPP_INFO(this->get_logger(), "Switching to Offboard mode");
            }
            
            if (offboard_setpoint_counter_ == 100) {
                arm();
            }
            return;
        }

        if (!has_arm_position_ || !has_attitude_data_ || !has_odometry_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "Missing data - arm:%d, attitude:%d, odom:%d",
                               has_arm_position_, has_attitude_data_, has_odometry_);
            return;
        }

        double angle_x = 0.0, angle_y = 0.0;
        bool has_deviation = false;
        {
            std::lock_guard<std::mutex> lock(deviation_mutex_);
            if (has_new_deviation_) {
                angle_x = last_angle_x_;
                angle_y = last_angle_y_;
                has_deviation = true;
                has_new_deviation_ = false;
            }
        }

        double current_relative_alt;
        {
            std::lock_guard<std::mutex> lock(odometry_mutex_);
            current_relative_alt = base_altitude_ - current_position_z_;
        }

        auto now = this->now();
        double takeoff_elapsed = (now - takeoff_start_time_).seconds();

        switch (current_state_) {
            case State::TAKEOFF:
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "Takeoff: current=%.1f m / target=%.1f m (elapsed=%.1f s)",
                                    current_relative_alt, takeoff_relative_altitude_, takeoff_elapsed);

                if (current_relative_alt >= takeoff_relative_altitude_ * 0.95) {
                    RCLCPP_INFO(this->get_logger(), "Takeoff complete (%.1f m)", current_relative_alt);
                    current_state_ = State::HOVER;

                    {
                        std::lock_guard<std::mutex> lock(odometry_mutex_);
                        hover_position_x_ = current_position_x_;
                        hover_position_y_ = current_position_y_;
                        hover_position_z_ = current_position_z_;
                    }
                }
                else if (takeoff_elapsed > takeoff_timeout_) {
                    RCLCPP_WARN(this->get_logger(), "Takeoff timeout after %.1f s", takeoff_elapsed);
                    current_state_ = State::HOVER;
                }
                break;

            case State::HOVER:
                if (has_deviation && !std::isnan(angle_x) && !std::isnan(angle_y)) {
                    RCLCPP_INFO(this->get_logger(), "Target detected, entering GUIDANCE");
                    current_state_ = State::GUIDANCE;
                    last_valid_angle_x_ = angle_x;
                    last_valid_angle_y_ = angle_y;
                    last_valid_target_time_ = now;
                    target_loss_count_ = 0;
                    has_valid_target_ = true;
                } else {
                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                       "Hovering: waiting for target (relative alt=%.1f m)",
                                       current_relative_alt);
                }
                break;

            case State::GUIDANCE:
                if (!has_valid_target_) {
                    RCLCPP_WARN(this->get_logger(), "No valid target history, returning to HOVER");
                    current_state_ = State::HOVER;
                    break;
                }
                
                rclcpp::Duration time_since_last = now - last_valid_target_time_;
                bool timeout = time_since_last.seconds() > target_loss_timeout_;
                bool count_exceeded = target_loss_count_ > target_loss_max_count_;

                if (timeout || count_exceeded) {
                    RCLCPP_WARN(this->get_logger(), "Target lost (timeout=%.2f s, count=%d), returning to HOVER",
                               time_since_last.seconds(), target_loss_count_);
                    current_state_ = State::HOVER;

                    {
                        std::lock_guard<std::mutex> lock(odometry_mutex_);
                        hover_position_x_ = current_position_x_;
                        hover_position_y_ = current_position_y_;
                        hover_position_z_ = current_position_z_;
                    }
                    break;
                } else {
                    if (has_deviation && !std::isnan(angle_x) && !std::isnan(angle_y)) {
                        proportional_guidance(angle_x, angle_y);
                    } else {
                        proportional_guidance(last_valid_angle_x_, last_valid_angle_y_);
                        target_loss_count_++;
                        
                        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 100,
                                           "Using last valid angles (%.2f, %.2f), loss count=%d",
                                           last_valid_angle_x_, last_valid_angle_y_, target_loss_count_);
                    }
                    enforce_altitude_constraints();
                }
                break;
        }
        
        // 发布控制命令
        publish_offboard_control_mode();
        publish_trajectory_setpoint();
    }
};

int main(int argc, char *argv[])
{
    std::cout << "Starting vision control..." << std::endl;
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<VisionControl>();
    rclcpp::spin(node);
    
    rclcpp::shutdown();
    return 0;
}