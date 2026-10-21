#pragma once

#include "ds41_moe.hpp"
#include "ds41_route_profile.hpp"
#include <functional>

namespace strata::v41 {

bool ds41_expert_devices(const std::string & selection, int visible,
                        std::vector<int> & devices, std::string & error);

// Dense/KV stays on its original device. Contiguous expert-layer shards own
// persistent VRAM caches and disjoint pinned tails; no file tier is allowed.
class Ds41MoeRunner : public Ds41MoeTier {
public:
    ~Ds41MoeRunner();
    bool init(const Ds41MoeGeom &, Ds41BlobSource *, const Ds41MoeConfig &,
              const std::vector<int> & devices, double reserve_gib, std::string & error,
              const std::vector<std::pair<int32_t,int32_t>> * selected = nullptr);
    Ds41RouteProfile route_profile;
    std::string route_profile_path;
    bool write_route_profile() const;
    bool sharded() const { return !shards_.empty(); }
    bool seed_from_ranked(const std::vector<std::pair<int32_t, int32_t>> &, std::string &);
    bool build_arena_ranked(const std::vector<std::pair<int32_t, int32_t>> &, std::string &);
    bool run(int64_t, const int32_t *, const float *, const float *, float *);
    bool run_to_dev(int64_t, const int32_t *, const float *, const float *, void *, const void * = nullptr);
    bool run_multi(int64_t, int, const int32_t *, const float *, const float *, float *);
    bool run_multi_to_dev(int64_t, int, const int32_t *, const float *, const float *, void *);
    bool run_chunk(int64_t, int, const int32_t *, const float *, const float *, float *);
    bool run_chunk_dev(int64_t, int, const int32_t *, const float *, const float *, void *);
    void release_cache();
    void release_chunk();
    void close();
    const Ds41MoeStats & stats() const;
    void reset_stats();
    const char * mode() const;
    int64_t resident() const;
    int64_t cache_slots() const;
    bool resident(int64_t, int64_t) const;
    bool in_arena(int64_t, int64_t) const;
    int64_t arena_experts() const;
    int64_t file_tier() const;
    double arena_gib() const;

private:
    struct Shard {
        int device = 0;
        std::unique_ptr<Ds41MoeTier> tier;
        void * output = nullptr;
        size_t output_bytes = 0;
    };
    bool dispatch(int64_t, const std::function<bool(Shard &)> &);
    bool run_device(int64_t, int, const int32_t *, const float *, const float *, void *, int, const void * = nullptr);
    std::vector<Shard> shards_;
    std::vector<size_t> owners_;
    int64_t hidden_ = 0;
    bool initialized_ = false;
    mutable Ds41MoeStats aggregate_;
};

} // namespace strata::v41
