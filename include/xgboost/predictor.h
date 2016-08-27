/*!
 * Copyright 2016 by Contributors
 * \file predictor.h
 * \brief alpha: Predictor for predicting margin of tree ensembles.
 * \author Tianqi Chen
 */
#ifndef XGBOOST_PREDICTOR_H_
#define XGBOOST_PREDICTOR_H_

#include <string>
#include "./base.h"
#include "./tree_model.h"

namespace xgboost {
/*!
 * \brief Predictor class that focused on doing prediction.
 *  Currently only implemented for regression tree ensemble.
 *
 * \code
 * RegTree::FVec vec;
 * vec.Init(predictor->NumFeature());
 * for (const SparseBatch::Inst& inst : mydata) {
 *   vec.Fill(inst);
 *   float margin = predictor->PredictMargin(vec);
 *   vec.Drop(inst);
 * }
 * \endcode
 */
class Predictor : public dmlc::Serializable {
 public:
  /*!
   * \brief Give the number of features this predictor uses.
   *  This can be used to initialize the temp structure RegTree::FVec
   * \return (maximum-feature-index + 1),
   */
  virtual uint32_t NumFeature() const = 0;
  /*!
   * \brief Predict margin score for one instance at a time
   *  For efficiency, this function returns margin score strictly.
   *  without applying transformation like logistic for binary:logistic
   *
   * \param vec the instance you want to predict.
   * \return The predicted margin score.
   */
  virtual float PredictMargin(const RegTree::FVec &vec) const = 0;
  /*!
   * \brief Predict margin score given a DMatrix.
   * \param dmat The DMatrix
   * \param preds The storage for prediction.
   */
  virtual void PredictMargin(DMatrix* data, std::vector<float>* preds) const = 0;
  /*!
   * \brief construct a tree ensemble predictor from set of trees.
   * \param trees The trees in the ensemble.
   * \param base_margin The base margin to be used in trees.
   */
  static Predictor* Create(const std::vector<const RegTree*> trees, float base_margin);
};
}  // namespace xgboost
#endif  // XGBOOST_PREDICTOR_H_
