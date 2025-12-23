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

inline double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

class VisionControl : public rclcpp::Node
{
public:
    VisionControl() : Node("virtual_control"),
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
                      target_velocity_z_(0.0),
                      virtual_vel_x_(0.0),
                      virtual_vel_y_(0.0),
                      virtual_vel_z_(0.0),
                      virtual_yaw_rate_(0.0)
    {
        declare_parameter("takeoff_relative_altitude", 45.0);
        declare_parameter("takeoff_timeout", 30.0);
        declare_parameter("proportional_gain", 4.0);
        declare_parameter("vertical_gain", 10.0);
        declare_parameter("yaw_gain", 1.0); 
        declare_parameter("max_speed", 5.0);
        declare_parameter("max_vertical_speed", 5.0);
        declare_parameter("max_yaw_rate", 0.5);
        declare_parameter("deviation_topic", "/camera/image_deviation"); 
        declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude");
        declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry");
        declare_parameter("min_relative_altitude", 0.5);
        declare_parameter("max_relative_altitude", 100.0);
        declare_parameter("target_loss_timeout", 0.5);
        declare_parameter("target_loss_max_count", 25);
        
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

        takeoff_start_time_ = this->now();

        offboard_control_mode_publisher_ = this->create_publisher<OffboardControlMode>(
            "/fmu/in/offboard_control_mode", 50);
        trajectory_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(
            "/fmu/in/trajectory_setpoint", 50);
        vehicle_command_publisher_ = this->create_publisher<VehicleCommand>(
            "/fmu/in/vehicle_command", 50);

        virtual_setpoint_publisher_ = this->create_publisher<TrajectorySetpoint>(
            "/vision_control/virtual_setpoint", 50);

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
        
        timer_ = this->create_wall_timer(20ms, std::bind(&VisionControl::timer_callback, this));
    }

private:
    enum class State { TAKEOFF, HOVER, GUIDANCE };
    
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Time takeoff_start_time_;
    rclcpp::Time last_valid_target_time_;
    rclcpp::Publisher<OffboardControlMode>::SharedPtr offboard_control_mode_publisher_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr trajectory_setpoint_publisher_;
    rclcpp::Publisher<VehicleCommand>::SharedPtr vehicle_command_publisher_;
    rclcpp::Subscription<qjwdds::msg::ImageDeviation>::SharedPtr image_deviation_sub_;
    rclcpp::Subscription<VehicleAttitude>::SharedPtr vehicle_attitude_sub_;
    rclcpp::Subscription<VehicleOdometry>::SharedPtr vehicle_odometry_sub_;
    rclcpp::Publisher<TrajectorySetpoint>::SharedPtr virtual_setpoint_publisher_;

    uint64_t offboard_setpoint_counter_;
    State current_state_;
    bool has_new_deviation_;
    bool has_attitude_data_;
    bool has_odometry_;
    bool has_arm_position_;
    bool has_valid_target_;
    std::mutex attitude_mutex_;
    std::array<float, 4> current_attitude_q_;
    double current_yaw_;
    double desired_yaw_;
    double takeoff_yaw_;
    std::mutex odometry_mutex_;
    double current_position_x_;
    double current_position_y_;
    double current_position_z_;
    std::mutex deviation_mutex_;
    double last_angle_x_;
    double last_angle_y_;
    double last_valid_angle_x_;
    double last_valid_angle_y_;
    int target_loss_count_;
    double arm_position_x_;
    double arm_position_y_;
    double base_altitude_;
    double hover_position_x_;
    double hover_position_y_;
    double hover_position_z_;
    double target_velocity_x_;
    double target_velocity_y_;
    double target_velocity_z_;

    double virtual_vel_x_;
    double virtual_vel_y_;
    double virtual_vel_z_;
    double virtual_yaw_rate_;

    double takeoff_relative_altitude_;
    double takeoff_timeout_;
    double proportional_gain_;
    double vertical_gain_;
    double yaw_gain_;
    double max_speed_;
    double max_vertical_speed_;
    double max_yaw_rate_;
    double min_relative_altitude_;
    double max_relative_altitude_;
    std::string deviation_topic_;
    std::string attitude_topic_;
    std::string odometry_topic_;
    double target_loss_timeout_;
    int target_loss_max_count_;

    double getYawFromQuaternion(const std::array<float, 4>& q) {
        double siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2]);
        double cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]);
        return std::atan2(siny_cosp, cosy_cosp);
    }

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
     * 虚拟引导逻辑: 计算指令但不更新实际控制变量
     * 偏航控制: 角速度控制 (Yaw Rate)
     */
    void compute_virtual_guidance(double angle_x_deg, double angle_y_deg) {
        double angle_x = angle_x_deg * M_PI / 180.0;
        double angle_y = angle_y_deg * M_PI / 180.0;

        double body_x = std::cos(angle_y) * std::cos(angle_x);
        double body_y = std::cos(angle_y) * std::sin(angle_x);
        double body_z = std::sin(angle_y);

        std::array<float, 4> current_q;
        {
            std::lock_guard<std::mutex> lock(attitude_mutex_);
            current_q = current_attitude_q_;
        }
        auto ground_vec = bodyToGround(current_q, {body_x, body_y, body_z});

        double des_vx = ground_vec[0] * proportional_gain_;
        double des_vy = ground_vec[1] * proportional_gain_;
        double des_vz = ground_vec[2] * vertical_gain_;

        double horizontal_speed = std::hypot(des_vx, des_vy);
        if (horizontal_speed > max_speed_) {
            double scale = max_speed_ / horizontal_speed;
            des_vx *= scale;
            des_vy *= scale;
        }
        if(des_vz > max_vertical_speed_) des_vz = max_vertical_speed_;
        else if (des_vz < -max_vertical_speed_) des_vz = -max_vertical_speed_;

        // 计算偏航角速度 (Yaw Rate)
        double des_yaw_rate = angle_x * yaw_gain_; 
        
        if (des_yaw_rate > max_yaw_rate_) des_yaw_rate = max_yaw_rate_;
        if (des_yaw_rate < -max_yaw_rate_) des_yaw_rate = -max_yaw_rate_;

        virtual_vel_x_ = des_vx;
        virtual_vel_y_ = des_vy;
        virtual_vel_z_ = des_vz;
        virtual_yaw_rate_ = des_yaw_rate;
    }

    // 发布虚拟指令
    void publish_virtual_command() {
        TrajectorySetpoint msg{};
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        msg.position = {NAN, NAN, NAN};
        msg.velocity = {(float)virtual_vel_x_, (float)virtual_vel_y_, (float)virtual_vel_z_};
        msg.yaw = NAN;
        msg.yawspeed = (float)virtual_yaw_rate_;
        virtual_setpoint_publisher_->publish(msg);
    }

    // 发布 Offboard 模式
    void publish_offboard_control_mode() {
        OffboardControlMode msg{};
        msg.position = true;
        msg.velocity = false;
        msg.acceleration = false;
        msg.attitude = false;
        msg.body_rate = false;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        offboard_control_mode_publisher_->publish(msg);
    }

    void publish_trajectory_setpoint() {
        TrajectorySetpoint msg{};
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;

        if (current_state_ == State::TAKEOFF) {
             msg.position[0] = arm_position_x_;
             msg.position[1] = arm_position_y_;
             msg.position[2] = base_altitude_ - takeoff_relative_altitude_;
        } else {
             msg.position[0] = hover_position_x_;
             msg.position[1] = hover_position_y_;
             msg.position[2] = hover_position_z_;
        }
        
        msg.velocity = {NAN, NAN, NAN};
        msg.yaw = (current_state_ == State::TAKEOFF) ? takeoff_yaw_ : current_yaw_; 
        msg.yawspeed = NAN;

        trajectory_setpoint_publisher_->publish(msg);
    }

    void publish_vehicle_command(uint16_t command, float param1 = 0.0, float param2 = 0.0) {
        VehicleCommand msg{};
        msg.param1 = param1; msg.param2 = param2; msg.command = command;
        msg.target_system = 1; msg.target_component = 1;
        msg.source_system = 1; msg.source_component = 1;
        msg.from_external = true;
        msg.timestamp = this->get_clock()->now().nanoseconds() / 1000;
        vehicle_command_publisher_->publish(msg);
    }

    void arm() {
        std::lock_guard<std::mutex> lock(odometry_mutex_);
        if (has_odometry_) {
            arm_position_x_ = current_position_x_;
            arm_position_y_ = current_position_y_;
            base_altitude_ = current_position_z_;
            has_arm_position_ = true;
            {
                std::lock_guard<std::mutex> lock(attitude_mutex_);
                if (has_attitude_data_) takeoff_yaw_ = current_yaw_;
            }
        }
        publish_vehicle_command(VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0);
    }

    void timer_callback() {
        if (offboard_setpoint_counter_ < 100) {
            publish_offboard_control_mode();
            publish_trajectory_setpoint();
            offboard_setpoint_counter_++;
            if (offboard_setpoint_counter_ == 50) publish_vehicle_command(VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);
            if (offboard_setpoint_counter_ == 100) arm();
            return;
        }

        if (!has_arm_position_ || !has_attitude_data_ || !has_odometry_) return;

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
                if (current_relative_alt >= takeoff_relative_altitude_ * 0.95 || takeoff_elapsed > takeoff_timeout_) {
                    RCLCPP_INFO(this->get_logger(), "Takeoff complete. Switching to HOVER.");
                    current_state_ = State::HOVER;
                    {
                        std::lock_guard<std::mutex> lock(odometry_mutex_);
                        hover_position_x_ = current_position_x_;
                        hover_position_y_ = current_position_y_;
                        hover_position_z_ = current_position_z_;
                    }
                }
                break;

            case State::HOVER:
                if (has_deviation && !std::isnan(angle_x) && !std::isnan(angle_y)) {
                    RCLCPP_INFO(this->get_logger(), "Target detected -> GUIDANCE (Virtual Mode)");
                    current_state_ = State::GUIDANCE;
                    last_valid_angle_x_ = angle_x;
                    last_valid_angle_y_ = angle_y;
                    last_valid_target_time_ = now;
                    target_loss_count_ = 0;
                    has_valid_target_ = true;
                }
                break;

            case State::GUIDANCE:
                if (!has_valid_target_) { current_state_ = State::HOVER; break; }
                
                {
                    rclcpp::Duration time_since_last = now - last_valid_target_time_;
                    bool timeout = time_since_last.seconds() > target_loss_timeout_;
                    if (timeout || target_loss_count_ > target_loss_max_count_) {
                         RCLCPP_WARN(this->get_logger(), "Target lost -> HOVER");
                         current_state_ = State::HOVER;
                         break;
                    }
                }

                if (has_deviation && !std::isnan(angle_x) && !std::isnan(angle_y)) {
                    compute_virtual_guidance(angle_x, angle_y);
                } else {
                    compute_virtual_guidance(last_valid_angle_x_, last_valid_angle_y_);
                    target_loss_count_++;
                }

                publish_virtual_command();
                break;
        }

        publish_offboard_control_mode();
        publish_trajectory_setpoint();
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionControl>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}