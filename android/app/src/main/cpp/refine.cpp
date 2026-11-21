#include "refine.h"

#include <android/log.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "stella_vslam/type.h"
#include "stella_vslam/data/graph_node.h"
#include "stella_vslam/data/keyframe.h"
#include "stella_vslam/data/landmark.h"
#include "stella_vslam/data/map_database.h"
#include "stella_vslam/module/loop_detector.h"
#include "stella_vslam/module/type.h"
#include "stella_vslam/optimize/graph_optimizer.h"
#include "stella_vslam/optimize/local_bundle_adjuster_g2o.h"
#include "stella_vslam/publish/map_publisher.h"
#include "stella_vslam/system.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "pocket-slam", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "pocket-slam", __VA_ARGS__)

namespace {

using keyframe = stella_vslam::data::keyframe;

std::vector<std::shared_ptr<keyframe>> live_keyframes(stella_vslam::system* slam) {
    std::vector<std::shared_ptr<keyframe>> keyframes;
    slam->get_map_publisher()->get_keyframes(keyframes);
    std::vector<std::shared_ptr<keyframe>> live;
    live.reserve(keyframes.size());
    for (const auto& keyframe : keyframes) {
        if (keyframe && !keyframe->will_be_erased()) live.push_back(keyframe);
    }
    return live;
}

double bow_score(const stella_vslam::data::bow_vector& left, const stella_vslam::data::bow_vector& right) {
    auto i = left.begin();
    auto j = right.begin();
    double score = 0.0;
    while (i != left.end() && j != right.end()) {
        if (i->first == j->first) {
            score += static_cast<double>(i->second) * static_cast<double>(j->second);
            ++i;
            ++j;
        } else if (i->first < j->first) {
            ++i;
        } else {
            ++j;
        }
    }
    return score;
}

g2o::Sim3 sim3_from_cw(const stella_vslam::Mat44_t& pose) {
    const stella_vslam::Mat33_t rotation = pose.block<3, 3>(0, 0);
    const stella_vslam::Vec3_t translation = pose.block<3, 1>(0, 3);
    return g2o::Sim3(rotation, translation, 1.0);
}

bool already_connected(const std::shared_ptr<keyframe>& query, const std::shared_ptr<keyframe>& other) {
    const auto neighbors = query->graph_node_->get_connected_keyframes();
    return neighbors.count(other) != 0;
}

void move_neighbor_landmarks(const stella_vslam::module::keyframe_Sim3_pairs_t& before,
                             const stella_vslam::module::keyframe_Sim3_pairs_t& after,
                             std::unordered_map<unsigned int, unsigned int>& ref_keyframe) {
    for (const auto& entry : after) {
        const auto& neighbor = entry.first;
        const g2o::Sim3 world_from_neighbor = entry.second.inverse();
        const g2o::Sim3& neighbor_from_world = before.at(neighbor);
        for (const auto& landmark : neighbor->get_landmarks()) {
            if (!landmark || landmark->will_be_erased()) continue;
            if (ref_keyframe.count(landmark->id_) != 0) continue;
            ref_keyframe[landmark->id_] = neighbor->id_;
            const stella_vslam::Vec3_t position = landmark->get_pos_in_world();
            landmark->set_pos_in_world(world_from_neighbor.map(neighbor_from_world.map(position)));
            landmark->update_mean_normal_and_obs_scale_variance();
        }
    }
}

bool apply_loop(const std::shared_ptr<keyframe>& current, const std::shared_ptr<keyframe>& candidate,
                const g2o::Sim3& world_to_current, stella_vslam::optimize::graph_optimizer& optimizer) {
    std::vector<std::shared_ptr<keyframe>> neighbors = current->graph_node_->get_covisibilities();
    neighbors.push_back(current);
    const stella_vslam::Mat44_t pose_before = current->get_pose_cw();
    stella_vslam::module::keyframe_Sim3_pairs_t before;
    stella_vslam::module::keyframe_Sim3_pairs_t after;
    for (const auto& neighbor : neighbors) {
        if (!neighbor || neighbor->will_be_erased() || neighbor->id_ == candidate->id_) continue;
        const stella_vslam::Mat44_t old_cw = neighbor->get_pose_cw();
        before[neighbor] = sim3_from_cw(old_cw);
        const stella_vslam::Mat44_t relative = old_cw * pose_before.inverse();
        const g2o::Sim3 relative_sim3(relative.block<3, 3>(0, 0), relative.block<3, 1>(0, 3), 1.0);
        after[neighbor] = relative_sim3 * world_to_current;
    }
    if (after.find(current) == after.end()) return false;
    std::unordered_map<unsigned int, unsigned int> ref_keyframe;
    {
        std::lock_guard<std::mutex> lock(stella_vslam::data::map_database::mtx_database_);
        move_neighbor_landmarks(before, after, ref_keyframe);
    }
    std::map<std::shared_ptr<keyframe>, std::set<std::shared_ptr<keyframe>>> connections;
    connections[current].insert(candidate);
    optimizer.optimize(candidate, current, before, after, connections, ref_keyframe);
    current->graph_node_->add_loop_edge(candidate);
    candidate->graph_node_->add_loop_edge(current);
    return true;
}

} // namespace

void pocket_freeze(stella_vslam::system* slam) {
    slam->disable_loop_detector();
    slam->disable_mapping_module();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (slam->loop_BA_is_running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

int pocket_local_ba(stella_vslam::system* slam, void (*tick)(int, int, void*), void* user) {
    const auto keyframes = live_keyframes(slam);
    const int total = static_cast<int>(keyframes.size());
    if (tick != nullptr) tick(0, total, user);
    if (total < 3) return 0;
    YAML::Node yaml;
    stella_vslam::data::map_database gauge(15);
    stella_vslam::optimize::local_bundle_adjuster_g2o adjuster(yaml);
    bool stop = false;
    int done = 0;
    int ran = 0;
    for (const auto& keyframe : keyframes) {
        ++done;
        if (tick != nullptr) tick(done, total, user);
        if (!keyframe || keyframe->will_be_erased() || keyframe->graph_node_->is_spanning_root()) continue;
        try {
            adjuster.optimize(&gauge, keyframe, &stop);
            ++ran;
        } catch (const std::exception& error) {
            LOGE("local BA %u: %s", keyframe->id_, error.what());
        }
    }
    LOGI("local BA ran on %d keyframes", ran);
    return ran;
}

int pocket_search_loops(stella_vslam::system* slam, void (*tick)(int, int, void*), void* user) {
    const auto keyframes = live_keyframes(slam);
    const int total = static_cast<int>(keyframes.size());
    if (tick != nullptr) tick(0, total, user);
    if (total < 8) return 0;
    YAML::Node yaml;
    stella_vslam::optimize::graph_optimizer optimizer(yaml, true);
    int accepted = 0;
    unsigned int quiet_until = 0;
    for (int index = 0; index < total; ++index) {
        if (tick != nullptr) tick(index + 1, total, user);
        const auto& current = keyframes[index];
        if (!current || current->will_be_erased() || current->id_ < quiet_until) continue;
        const auto covisibilities = current->graph_node_->get_covisibilities();
        double floor_score = 1.0;
        for (const auto& neighbor : covisibilities) {
            if (!neighbor || neighbor->will_be_erased()) continue;
            floor_score = std::min(floor_score, bow_score(current->bow_vec_, neighbor->bow_vec_));
        }
        std::shared_ptr<keyframe> best;
        double best_score = floor_score;
        for (const auto& other : keyframes) {
            if (!other || other->id_ == current->id_ || other->will_be_erased()) continue;
            if (current->id_ < other->id_ + 20 && other->id_ < current->id_ + 20) continue;
            if (already_connected(current, other)) continue;
            const double score = bow_score(current->bow_vec_, other->bow_vec_);
            if (score > best_score) {
                best_score = score;
                best = other;
            }
        }
        if (!best) continue;
        try {
            stella_vslam::module::loop_detector detector(nullptr, nullptr, yaml, true);
            detector.set_current_keyframe(current);
            detector.add_loop_candidate(best);
            if (!detector.validate_candidates()) continue;
            const auto selected = detector.get_selected_candidate_keyframe();
            if (!selected) continue;
            if (!apply_loop(current, selected, detector.get_Sim3_world_to_current(), optimizer)) continue;
            ++accepted;
            quiet_until = current->id_ + 10;
            LOGI("loop %u - %u score %.3f", selected->id_, current->id_, best_score);
        } catch (const std::exception& error) {
            LOGE("loop %u: %s", current->id_, error.what());
        }
    }
    LOGI("accepted %d loops", accepted);
    return accepted;
}

void pocket_resume(stella_vslam::system* slam) {
    slam->enable_loop_detector();
    slam->enable_mapping_module();
}
