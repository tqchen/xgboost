/*!
 * Copyright 2016 by Contributors
 * \file hist_data.h
 * \brief Utility for fast histogram aggregation
 * \author Tianqi Chen
 */
#ifndef XGBOOST_COMMON_HIST_UTIL_H_
#define XGBOOST_COMMON_HIST_UTIL_H_

#include <xgboost/data.h>

namespace xgboost {
namespace common {

/*! \brief Cut configuration for one feature */
struct HistCutUnit {
  /*! \brief the index pointer of each histunit */
  const bst_float *cut;
  /*! \brief number of cutting point, containing the maximum point */
  size_t size;
  // default constructor
  HistCutUnit() {}
  // constructor
  HistCutUnit(const bst_float *cut, unsigned size)
      : cut(cut), size(size) {}
};

/*! \brief cut configuration for all the features */
struct HistCutMatrix {
  /*! \brief actual unit pointer */
  std::vector<unsigned> row_ptr;
  /*! \brief the cut field */
  std::vector<bst_float> cut;
  /*! \brief Get histogram bound for fid */
  inline HistCutUnit operator[](unsigned fid) const {
    return HistCutUnit(dmlc::BeginPtr(cut) + row_ptr[fid],
                       row_ptr[fid + 1] - row_ptr[fid]);
  }
  // create histogram cut matrix given statistics from data
  // using approximate quantile sketch approach
  void Init(DMatrix *p_fmat, size_t max_num_bins);
};


/*!
 * \brief A single row in global histogram index.
 *  Directly represent the global index in the histogram entry.
 */
struct GHistIndexRow {
  /*! \brief The index of the histogram */
  const unsigned *index;
  /*! \brief The size of the histogram */
  unsigned size;
  GHistIndexRow() {}
  GHistIndexRow(const unsigned* index, unsigned size)
      : index(index), size(size) {}
};

/*!
 * \brief preprocessed global index matrix, in CSR format
 *  Transform floating values to integer index in histogram
 *  This is a global histogram hindex.
 */
struct GHistIndexMatrix {
  /*! \brief row pointer */
  std::vector<unsigned> row_ptr;
  /*! \brief The index data */
  std::vector<unsigned> index;
  /*! \brief hit count of each index */
  std::vector<unsigned> hit_count;
  /*! \brief optional remap index from outter row_id -> internal row_id*/
  std::vector<unsigned> remap_index;
  /*! \brief The corresponding cuts */
  const HistCutMatrix* cut;
  // Create a global histogram matrix, given cut
  void Init(DMatrix *p_fmat);
  // build remap
  void Remap();
  // get i-th row
  inline GHistIndexRow operator[](bst_uint i) const {
    return GHistIndexRow(
        &index[0] + row_ptr[i], row_ptr[i + 1] - row_ptr[i]);
  }
};


// run a histogram aggregation
void MakeHist(const std::vector<bst_gpair>& gpair,
              const std::vector<bst_uint>& row_indices,
              const GHistIndexMatrix& gmat,
              std::vector<bst_gpair>* hist);


}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_HIST_DATA_H_
