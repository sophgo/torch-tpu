/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "sophon/algorithm.h"
#include "sophon/context.h"

namespace sophon {

template <typename T>
class AllreduceLocal : public Algorithm {
 public:
  AllreduceLocal(
      const std::shared_ptr<Context>& context,
      const std::vector<T*>& ptrs,
      const int count,
      const ReductionFunction<T>* fn = ReductionFunction<T>::sum);

  virtual ~AllreduceLocal() = default;

  virtual void run() override;

 protected:
  std::vector<T*> ptrs_;
  const int count_;
  const int bytes_;
  const ReductionFunction<T>* fn_;
};

} // namespace sophon
