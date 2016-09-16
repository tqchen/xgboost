/*!
 * Copyright 2014 by Contributors
 * \file updater_refresh.cc
 * \brief refresh the statistics and leaf value on the tree on the dataset
 * \author Tianqi Chen
 */

#include <xgboost/tree_updater.h>
#include <vector>
#include <limits>
#include "./param.h"
#include "../common/sync.h"
#include "../common/io.h"

namespace xgboost {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_fully_correct);

/*! \brief A corrector that corrects the leaf value */
template<typename TStats>
class TreeCorrector: public TreeUpdater {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& args) override {
    param.InitAllowUnknown(args);
  }
  // update the tree, do pruning
  void Update(const std::vector<bst_gpair> &gpair,
              DMatrix *p_fmat,
              const std::vector<RegTree*> &trees) {
    CHECK_EQ(trees.size(), 1);
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    fvec_temp_.resize(nthread, RegTree::FVec());
    stemp_.resize(nthread, std::vector<TStats>());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int num_nodes = 0;
      for (size_t i = 0; i < trees.size(); ++i) {
        num_nodes += trees[i]->param.num_nodes;
      }
      stemp_[tid].resize(num_nodes, TStats(param));
      std::fill(stemp_[tid].begin(), stemp_[tid].end(), TStats(param));
      fvec_temp_[tid].Init(trees[0]->param.num_feature);
    }
    // if it is C++11, use lazy evaluation for Allreduce,
    // to gain speedup in recovery

    const MetaInfo &info = p_fmat->info();
    leaf_index_.resize(info.num_row);
    // start accumulating statistics
    dmlc::DataIter<RowBatch> *iter = p_fmat->RowIterator();
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch &batch = iter->Value();
      CHECK_LT(batch.size, std::numeric_limits<unsigned>::max());
      const bst_omp_uint nbatch = static_cast<bst_omp_uint>(batch.size);
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < nbatch; ++i) {
        RowBatch::Inst inst = batch[i];
        const int tid = omp_get_thread_num();
        const bst_uint ridx = static_cast<bst_uint>(batch.base_rowid + i);
        RegTree::FVec &feats = fvec_temp_[tid];
        feats.Fill(inst);
        int pid = AddStats(*trees[0], feats, gpair, info, ridx,
                           dmlc::BeginPtr(stemp_[tid]), true);
        leaf_index_[ridx] = pid;
        feats.Drop(inst);
      }
    }
    // aggregate the statistics
    int num_nodes = static_cast<int>(stemp_[0].size());
    #pragma omp parallel for schedule(static)
    for (int nid = 0; nid < num_nodes; ++nid) {
      for (int tid = 1; tid < nthread; ++tid) {
        stemp_[0][nid].Add(stemp_[tid][nid]);
      }
    }
    reducer.Allreduce(dmlc::BeginPtr(stemp_[0]), stemp_[0].size());
    // rescale learning rate according to size of trees
    for (int rid = 0; rid < trees[0]->param.num_roots; ++rid) {
      this->Refresh(dmlc::BeginPtr(stemp_[0]) , rid, trees[0]);
    }
  }
  const int* GetLeafPosition() const override {
    return dmlc::BeginPtr(leaf_index_);
  }
 private:
  inline static int AddStats(const RegTree &tree,
                             const RegTree::FVec &feat,
                             const std::vector<bst_gpair> &gpair,
                             const MetaInfo &info,
                             const bst_uint ridx,
                             TStats *gstats,
                             bool update_leaf_only) {
    // start from groups that belongs to current data
    int pid = static_cast<int>(info.GetRoot(ridx));
    // tranverse tree
    while (!tree[pid].is_leaf()) {
      if (!update_leaf_only) {
        gstats[pid].Add(gpair, info, ridx);
      }
      unsigned split_index = tree[pid].split_index();
      pid = tree.GetNext(pid, feat.fvalue(split_index), feat.is_missing(split_index));
    }
    gstats[pid].Add(gpair, info, ridx);
    return pid;
  }
  inline void Refresh(const TStats *gstats,
                      int nid, RegTree *p_tree) {
    RegTree &tree = *p_tree;
    for (size_t nid = 0; nid < tree.param.num_nodes; ++nid) {
      if (tree[nid].is_leaf() && !tree[nid].is_deleted()) {
        TStats s = gstats[nid];
        double w = tree[nid].leaf_value();
        double dw = param.CalcDelta(s.sum_grad, s.sum_hess, w);
        tree[nid].set_leaf(w + dw * param.learning_rate);
      }
    }
  }
  // training parameter
  TrainParam param;
  // thread temporal storage
  std::vector<std::vector<TStats> > stemp_;
  // thread temporal storage
  std::vector<RegTree::FVec> fvec_temp_;
  // leaf index
  std::vector<int> leaf_index_;
  // reducer
  rabit::Reducer<TStats, TStats::Reduce> reducer;
};

XGBOOST_REGISTER_TREE_UPDATER(TreeCorrector, "correct")
.describe("Corrector that correct the leaf weight according to gradient statistics.")
.set_body([]() {
    return new TreeCorrector<GradStats>();
  });
}  // namespace tree
}  // namespace xgboost
