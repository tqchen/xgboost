/*!
 * Copyright 2015 by Contributors
 * \file tree_ensemble.cc
 * \brief Implementation of tree ensemble predictor.
 */
#include <utility>
#include <dmlc/logging.h>
#include <dmlc/timer.h>
#include <xgboost/predictor.h>

namespace xgboost {
namespace pred {
/*! \brief type of index in the tree */
typedef uint32_t tindex_t;

/*! \brief tree node optimized for memory access */
struct TreeNode {
  /*! \brief the feature index we are looking at */
  uint32_t sindex_;
  /*! \brief split value or leaf value */
  float value_;
  /*! \brief index of childrens */
  tindex_t children_[2];
  // default constructor
  /*! \brief feature index of split condition */
  inline uint32_t split_index() const {
    return sindex_ & ((1U << 31) - 1U);
  }
  /*! \brief when feature is unknown, whether goes to left child */
  inline int default_left() const {
    return (sindex_ >> 31);
  }
  /*!
   * \brief set split condition of current node
   * \param split_index feature index to split
   * \param split_value  split condition
   * \param cleft left child
   * \param cright right child
   * \param default_left the default direction when feature is unknown
   */
  inline void set_split(uint32_t split_index,
                        float split_value,
                        tindex_t cleft,
                        tindex_t cright,
                        bool default_left) {
    if (default_left) split_index |= (1U << 31);
    sindex_ = split_index;
    value_ = split_value;
    children_[0] = cleft;
    children_[1] = cright;
  }
  /*!
   * \brief set the node to be leaf
   * \param leaf_value the value to set
   */
  inline void set_leaf(float leaf_value) {
    sindex_ = 0;
    value_ = leaf_value;
    children_[0] = children_[1] = 0;
  }
  /*! \return if the node is leaf */
  inline bool is_leaf() const {
    return cleft() == 0;
  }
  /*! \return left index */
  inline tindex_t cleft() const {
    return children_[0];
  }
  /*! \return right index */
  inline tindex_t cright() const {
    return children_[1];
  }
  /*! \return split value */
  inline float split_value() const {
    return value_;
  }
  /*! \return split value */
  inline float leaf_value() const {
    return value_;
  }
};

class TreeEnsemblePredictor : public Predictor {
 public:
  uint32_t NumFeature() const override {
    return max_split_index_ + 1;
  }

  void PredictMargin(DMatrix* data, std::vector<float>* out_preds) const override {
    this->Benchmark(data, out_preds);
  }

  float PredictMargin(const RegTree::FVec& vec) const override {
    return PredictMarginInternal(vec) + base_margin_;
  }
  void Load(dmlc::Stream* fi) override {
    LOG(FATAL) << "not implemented";
  }
  void Save(dmlc::Stream* fo) const override {
    LOG(FATAL) << "not implemented";
  }
  // constructor
  static Predictor* Create(const std::vector<const RegTree*> trees, float base_margin);

 private:
  // internal constructor
  TreeEnsemblePredictor(std::vector<TreeNode>&& nodes,
                        std::vector<uint32_t>&& roots,
                        float base_margin)
      : nodes_(std::move(nodes)), roots_(std::move(roots)), base_margin_(base_margin) {
    this->Init();
  }
  // initialize internal states
  inline void Init() {
    max_split_index_ = 0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
      const TreeNode& node = nodes_[i];
      if (!node.is_leaf() && max_split_index_ <= node.split_index()) {
        max_split_index_ = node.split_index();
      }
    }
  }
  inline float PredictMarginInternal(const RegTree::FVec& vec) const {
    CHECK_GE(vec.Size(), max_split_index_);
    return PredictMarginSimple(vec);
  }

  void Benchmark(DMatrix* data, std::vector<float>* out_preds) const {
    int nrep = 1;
    Benchmark_(data, out_preds, nrep, [this](const RegTree::FVec& v) {
        return this->PredictMarginSimple(v);
      }, "simple");
    Benchmark_(data, out_preds, nrep, [this](const RegTree::FVec& v) {
        return this->PredictMarginTable(v);
      }, "table");
    Benchmark_(data, out_preds, nrep, [this](const RegTree::FVec& v) {
        return this->PredictMarginUnroll(v);
      }, "unroll");
    Benchmark_(data, out_preds, nrep, [this](const RegTree::FVec& v) {
        return this->PredictMarginUnrollX(v);
      }, "unrollx");
  }

  template<typename TPred>
  inline void Benchmark_(DMatrix* data, std::vector<float>* out_preds,
                         int nrep,
                         TPred pfun, const char *msg) const {
    double tstart = dmlc::GetTime();
    RegTree::FVec tvec;
    tvec.Init(this->NumFeature());

    const std::vector<bst_float>& base_margin = data->info().base_margin;
    std::vector<float> &preds = *out_preds;
    preds.resize(data->info().num_row);
    dmlc::DataIter<RowBatch>* iter = data->RowIterator();
    double tsum = 0.0;
    iter->BeforeFirst();
    while (iter->Next()) {
      const RowBatch& batch = iter->Value();
      double tic = dmlc::GetTime();
      for (int i = 0; i < nrep; ++i) {
        for (size_t i = 0; i < batch.size; ++i) {
          tvec.Fill(batch[i]);
          float sum = pfun(tvec);
          tvec.Drop(batch[i]);
          preds[batch.base_rowid + i] = sum;
        }
      }
      double toc = dmlc::GetTime();
      tsum += toc - tic;
    }

    for (size_t i = 0; i < preds.size(); ++i) {
      if (base_margin.size() != 0) {
        preds[i] += base_margin[i];
      } else {
        preds[i] += base_margin_;
      }
    }

    double total = dmlc::GetTime() - tstart;
    LOG(INFO) << msg << ":: tsum=" << tsum << " ttotal=" << total;
  }

  // predict the margin
  inline float PredictMarginSimple(const RegTree::FVec& vec) const {
    float sum = 0.0f;
    for (size_t i = 0; i < roots_.size(); ++i) {
      uint32_t pid = roots_[i];
      while (!nodes_[pid].is_leaf()) {
        const TreeNode& n = nodes_[pid];
        uint32_t split_index = n.split_index();
        if (vec.is_missing(split_index)) {
          pid = n.default_left() ? n.cleft() : n.cright();
        } else {
          pid = vec.fvalue(split_index) < n.split_value() ? n.cleft() : n.cright();
        }
      }
      sum += nodes_[pid].leaf_value();
    }
    return sum;
  }

  inline float PredictMarginTable(const RegTree::FVec& vec) const {
    float sum = 0.0f;
    for (size_t i = 0; i < roots_.size(); ++i) {
      uint32_t pid = roots_[i];
      while (!nodes_[pid].is_leaf()) {
        const TreeNode& n = nodes_[pid];
        uint32_t split_index = n.split_index();
        RegTree::FVec::Entry e = vec.data[split_index];
        pid = n.children_[e.fvalue >= n.split_value()];
      }
      sum += nodes_[pid].leaf_value();
    }
    return sum;
  }

  inline float PredictMarginUnroll(const RegTree::FVec& vec) const {
    double sum = 0.0f;
    const int K = 10;
    size_t root_counter = 0;
    tindex_t buf_index[K];
    TreeNode buf_node[K];
    RegTree::FVec::Entry buf_entry[K];

    for (int i = 0; i < K; ++i) {
      buf_index[i] = 0;
    }

    while (true) {
      // fetch node
      for (int i = 0; i < K; ++i) {
        buf_node[i] = nodes_[buf_index[i]];
      }
      // get entry
      for (int i = 0; i < K; ++i) {
        buf_entry[i] = vec.data[buf_node[i].split_index()];
      }
      // get condition
      for (int i = 0; i < K; ++i) {
        RegTree::FVec::Entry e = buf_entry[i];
        auto& n = buf_node[i];
        if (!n.is_leaf()) {
          if (e.flag != -1) {
            buf_index[i] = n.children_[e.fvalue >= n.split_value()];
          } else {
            buf_index[i] = n.children_[!n.default_left()];
          }
        } else {
          sum += n.leaf_value();
          buf_index[i] = roots_[root_counter++];
          // tail handling.
          if (root_counter == roots_.size()) {
            for (int j = 0; j < K; ++j) {
              sum += GetLeafValue(buf_index[j], vec);
            }
            return sum;
          }
        }
      }
    }
  }

  inline float PredictMarginUnrollX(const RegTree::FVec& vec) const {
    double sum = 0.0f;
    size_t root_counter = 0;
    tindex_t i0, i1, i2, i3;
    TreeNode n0, n1, n2, n3;
    RegTree::FVec::Entry e0, e1, e2, e3;

    i0 = i1 = i2 = i3 = 0;

    while (true) {
      // fetch node
      n0 = nodes_[i0];
      n1 = nodes_[i1];
      n2 = nodes_[i2];
      n3 = nodes_[i3];

      e0 = vec.data[n0.split_index()];
      e1 = vec.data[n1.split_index()];
      e2 = vec.data[n2.split_index()];
      e3 = vec.data[n3.split_index()];

      if (!n0.is_leaf()) {
        if (e0.flag != -1) {
          i0 = n0.children_[e0.fvalue >= n0.split_value()];
        } else {
          i0 = n0.children_[!n0.default_left()];
        }
      } else {
        sum += n0.leaf_value();
        i0 = roots_[root_counter++];
        // tail handling.
        if (root_counter == roots_.size()) {
          sum += GetLeafValue(i0, vec);
          sum += GetLeafValue(i1, vec);
          sum += GetLeafValue(i2, vec);
          sum += GetLeafValue(i3, vec);
          return sum;
        }
      }


      if (!n1.is_leaf()) {
        if (e1.flag != -1) {
          i1 = n1.children_[e1.fvalue >= n1.split_value()];
        } else {
          i1 = n1.children_[!n1.default_left()];
        }
      } else {
        sum += n1.leaf_value();
        i1 = roots_[root_counter++];
        // tail handling.
        if (root_counter == roots_.size()) {
          sum += GetLeafValue(i0, vec);
          sum += GetLeafValue(i1, vec);
          sum += GetLeafValue(i2, vec);
          sum += GetLeafValue(i3, vec);
          return sum;
        }
      }

      if (!n2.is_leaf()) {
        if (e2.flag != -1) {
          i2 = n2.children_[e2.fvalue >= n2.split_value()];
        } else {
          i2 = n2.children_[!n2.default_left()];
        }
      } else {
        sum += n2.leaf_value();
        i2 = roots_[root_counter++];
        // tail handling.
        if (root_counter == roots_.size()) {
          sum += GetLeafValue(i0, vec);
          sum += GetLeafValue(i1, vec);
          sum += GetLeafValue(i2, vec);
          sum += GetLeafValue(i3, vec);
          return sum;
        }
      }

      if (!n3.is_leaf()) {
        if (e3.flag != -1) {
          i3 = n3.children_[e3.fvalue >= n3.split_value()];
        } else {
          i3 = n3.children_[!n3.default_left()];
        }
      } else {
        sum += n3.leaf_value();
        i3 = roots_[root_counter++];
        // tail handling.
        if (root_counter == roots_.size()) {
          sum += GetLeafValue(i0, vec);
          sum += GetLeafValue(i1, vec);
          sum += GetLeafValue(i2, vec);
          sum += GetLeafValue(i3, vec);
          return sum;
        }
      }
    }
  }

  float GetLeafValue(tindex_t pid, const RegTree::FVec& vec) const {
    while (!nodes_[pid].is_leaf()) {
      const TreeNode& n = nodes_[pid];
      uint32_t split_index = n.split_index();
      if (vec.is_missing(split_index)) {
        pid = n.default_left() ? n.cleft() : n.cright();
      } else {
        pid = vec.fvalue(split_index) < n.split_value() ? n.cleft() : n.cright();
      }
    }
    return nodes_[pid].leaf_value();
  }



  // nodes of all tree ensembles
  std::vector<TreeNode> nodes_;
  // root index of each tree in the ensemble
  std::vector<uint32_t> roots_;
  // base margin
  float base_margin_;
  // maximum split index in the nodes
  uint32_t max_split_index_;
};

Predictor* TreeEnsemblePredictor::Create(
    const std::vector<const RegTree*> trees, float base_margin) {
  // nodes of all tree ensembles
  std::vector<TreeNode> nodes(1);
  std::vector<tindex_t> roots;
  // reserve node one
  nodes[0].set_leaf(0.0f);

  for (const RegTree* ptree : trees) {
    const RegTree& old_tree = *ptree;
    CHECK_EQ(old_tree.param.num_roots, 1);
    std::vector<std::pair<int, tindex_t> > stack;
    CHECK_GE(static_cast<size_t>(std::numeric_limits<tindex_t>::max()), nodes.size());
    tindex_t new_root_id = static_cast<tindex_t>(nodes.size());
    nodes.resize(nodes.size() + 1);
    roots.push_back(new_root_id);
    stack.push_back(std::make_pair(0, new_root_id));

    while (!stack.empty()) {
      std::pair<int, tindex_t> p = stack.back();
      stack.pop_back();
      const RegTree::Node& old_node = old_tree[p.first];
      tindex_t new_id = p.second;
      if (old_node.is_leaf()) {
        nodes[new_id].set_leaf(old_node.leaf_value());
        CHECK(nodes[new_id].is_leaf());
      } else {
        tindex_t new_left_id = static_cast<tindex_t>(nodes.size());
        tindex_t new_right_id = static_cast<tindex_t>(nodes.size() + 1);
        nodes.resize(nodes.size() + 2);
        nodes[new_id].set_split(old_node.split_index(),
                                old_node.split_cond(),
                                new_left_id,
                                new_right_id,
                                old_node.default_left());
        CHECK_LT(new_id, new_left_id);
        CHECK_LT(new_id, new_right_id);
        stack.push_back(std::make_pair(old_node.cright(), new_right_id));
        stack.push_back(std::make_pair(old_node.cleft(), new_left_id));
      }
    }
  }

  return new TreeEnsemblePredictor(
      std::move(nodes), std::move(roots), base_margin);
}
}  // namespace pred

Predictor* Predictor::Create(
    const std::vector<const RegTree*> trees, float base_margin) {
  return pred::TreeEnsemblePredictor::Create(trees, base_margin);
}
}  // namespace xgboost
