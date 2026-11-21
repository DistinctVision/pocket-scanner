#pragma once

namespace stella_vslam {
class system;
}

// Drain the mapper, then keep it paused so a later local BA is not aborted.
void pocket_freeze(stella_vslam::system* slam);

// Run stella's local bundle adjustment on every keyframe. Returns how many ran.
int pocket_local_ba(stella_vslam::system* slam, void (*tick)(int done, int total, void* user), void* user);

// BoW candidates, then stella's geometric check. Only an accepted pair moves the graph.
// Returns how many loops were applied.
int pocket_search_loops(stella_vslam::system* slam, void (*tick)(int done, int total, void* user), void* user);

void pocket_resume(stella_vslam::system* slam);
